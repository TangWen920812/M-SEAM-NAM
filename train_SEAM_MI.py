
import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
import covid19.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F

def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='none')
    if gamma == 0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss
    weighted_loss = alpha * loss

    return torch.mean(weighted_loss)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=16, type=int)
    parser.add_argument("--network", default="network.resnet38_SEAM_MI", type=str)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="covid19/train_MI.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="resnet38_SEAM_MI", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='/mnt/1f4e1319-2148-41a1-b7a3-8119cd25558c/', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))
    
    model = getattr(importlib.import_module(args.network), 'Net')()

    print(model)

    tblogger = SummaryWriter(args.tblog_dir)	

    # train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
    #                                            transform=transforms.Compose([
    #                     imutils.RandomResizeLong(448, 768),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #                     np.asarray,
    #                     model.normalize,
    #                     imutils.RandomCrop(args.crop_size),
    #                     imutils.HWC_to_CHW,
    #                     torch.from_numpy
    #                 ]))
    train_dataset = covid19.data.DicomDatasetMI(
                        txt_file=args.train_list, dicom_root=args.voc12_root,
                        transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 448),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        # transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01), # 0.3,0.3,0.3,0.1
                        np.asarray,
                        model.normalize,
                        imutils.CenterCrop(args.crop_size), # RandomCrop
                        imutils.HWC_to_CHW,
                        # torch.from_numpy
                    ]))
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=1,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)

    max_step = 1100*300*args.max_epoches
    # max_step = 0
    # for iter, pid_pack in enumerate(train_data_loader):
    #     max_step += pid_pack[1].shape[1]
    # max_step = max_step // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([          #PolyOptimizer AdamOptimizer
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0}, # 2
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec}, # 10
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0} # 20
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)  # betas=[0.9, 0.999]

    if args.weights[-7:] == '.params':
        import network.resnet38d
        assert 'resnet38' in args.network
        weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'bg_loss', 'fg_loss', 'neg_loss',
                                     'loss_rvmin')

    timer = pyutils.Timer("Session started: ")
    extract_aff_lab_func = voc12.data.ExtractAffinityLabelInRadius(cropsize=448 // 8, radius=5)
    for ep in range(args.max_epoches):
        loss_cls = 0
        for iter, pid_pack in enumerate(train_data_loader):
            label = pid_pack[2]
            bg_score = torch.ones((1, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            pred_list1, pred_list2 = [], []

            for i in range(0, pid_pack[1].shape[1], args.batch_size):

                pack = pid_pack[1][0, i:i+args.batch_size]
                scale_factor = 0.3
                img1 = pack
                img2 = F.interpolate(img1,scale_factor=scale_factor,mode='bilinear',align_corners=True)
                N,C,H,W = img1.size()

                cam1, cam_rv1, aff1 = model(img1, aff=True)
                label1 = F.adaptive_avg_pool2d(cam1, (1,1))
                loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
                #todo multiplied with label or not

                label_aff = F.interpolate(cam_rv1, (H//8, W//8), mode='bilinear', align_corners=True).detach().cpu().numpy()
                label_aff2 = label_aff.copy()
                label_aff[label_aff < 0.2] = 0
                label_aff[label_aff >= 0.2] = 1
                label_aff[:, 0, :, :] = 1 - label_aff[:, 1, :, :]
                label_aff2[label_aff2 < 0.01] = 0
                label_aff2[label_aff2 >= 0.01] = 1
                label_aff2[:, 0, :, :] = 1 - label_aff2[:, 1, :, :]

                label_la, label_ha = label_aff.transpose((0,2,3,1)), label_aff2.transpose((0,2,3,1))
                label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
                label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
                label_a = label_la.copy()
                label_a[label_la == 0] = 255
                label_a[label_ha == 0] = 0
                bg_label, fg_label, neg_label = [], [], []
                for i in range(label_a.shape[0]):
                    bg_pos_affinity_label, fg_pos_affinity_label, neg_affinity_label = extract_aff_lab_func(label_a[i])
                    bg_label.append(bg_pos_affinity_label)
                    fg_label.append(fg_pos_affinity_label)
                    neg_label.append(neg_affinity_label)

                aff1 = aff1.view(-1, aff1.size(-1))
                bg_label = torch.cat(bg_label, dim=0).cuda(non_blocking=True)
                fg_label = torch.cat(fg_label, dim=0).cuda(non_blocking=True)
                neg_label = torch.cat(neg_label, dim=0).cuda(non_blocking=True)
                bg_count = torch.sum(bg_label) + 1e-5
                fg_count = torch.sum(fg_label) + 1e-5
                neg_count = torch.sum(neg_label) + 1e-5
                bg_loss = torch.sum(- bg_label * torch.log(aff1 + 1e-5)) / bg_count
                fg_loss = torch.sum(- fg_label * torch.log(aff1 + 1e-5)) / fg_count
                neg_loss = torch.sum(- neg_label * torch.log(1. + 1e-5 - aff1)) / neg_count

                cam1 = F.interpolate(visualization.max_norm(cam1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
                cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label


                cam2, cam_rv2 = model(img2, aff=False)
                label2 = F.adaptive_avg_pool2d(cam2, (1,1))
                loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
                cam2 = visualization.max_norm(cam2)*label
                cam_rv2 = visualization.max_norm(cam_rv2)*label

                pred_list1.append(label1[:,1:,:,:])
                pred_list2.append(label2[:,1:,:,:])
                loss_cls1 = F.soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
                loss_cls2 = F.soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])
                # loss_cls1 = focal_loss(label1[:, 1:, :, :], label[:, 1:, :, :], 1, 2)
                # loss_cls2 = focal_loss(label2[:, 1:, :, :], label[:, 1:, :, :], 1, 2)

                ns,cs,hs,ws = cam2.size()
                loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
                #loss_er = torch.mean(torch.pow(cam1[:,1:,:,:]-cam2[:,1:,:,:], 2))
                cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
                cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]
    #            with torch.no_grad():
    #                eq_mask = (torch.max(torch.abs(cam1-cam2),dim=1,keepdim=True)[0]<0.7).float()
                tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)#*eq_mask
                tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)#*eq_mask
                loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(2*hs*ws*0.2), dim=-1)[0])
                loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(2*hs*ws*0.2), dim=-1)[0])
                loss_ecr = loss_ecr1 + loss_ecr2

                loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2 * 1
                loss_rvmin = (loss_rvmin1 + loss_rvmin2)/2 * 1
                loss = loss_rvmin + loss_er * 1 + loss_ecr * 1 + (bg_loss/40 + fg_loss/40 + neg_loss/20) * min(ep + 1, 10)
                loss = loss + loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_meter.add({'loss': loss.item(), 'loss_rvmin': loss_rvmin.item(), 'loss_er': loss_er.item(), 'loss_ecr': loss_ecr.item(),
                               'bg_loss': bg_loss.item(), 'fg_loss': fg_loss.item(), 'neg_loss': neg_loss.item(),
                               'loss_cls': loss_cls if type(loss_cls) == int else loss_cls.item()})

                if (optimizer.global_step - 1) % 10 == 0:

                    timer.update_progress(optimizer.global_step / max_step)

                    print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                          'loss:%.4f %.4f %.4f %.4f %.4f %.4f %.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er',
                                                                          'loss_ecr', 'bg_loss', 'fg_loss', 'neg_loss'),
                          'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                          'Fin:%s' % (timer.str_est_finish()),
                          'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                    avg_meter.pop()

                    # Visualization for training process
                    img_8 = img1[0].numpy().transpose((1,2,0))
                    img_8 = np.ascontiguousarray(img_8)
                    mean = (0.485, 0.456, 0.406)
                    std = (0.229, 0.224, 0.225)
                    img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                    img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                    img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                    img_8[img_8 > 255] = 255
                    img_8[img_8 < 0] = 0
                    img_8 = img_8.astype(np.uint8)

                    input_img = img_8.transpose((2,0,1))
                    h = H//4; w = W//4
                    p1 = F.interpolate(cam1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                    p2 = F.interpolate(cam2,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                    p_rv1 = F.interpolate(cam_rv1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                    p_rv2 = F.interpolate(cam_rv2,(h,w),mode='bilinear')[0].detach().cpu().numpy()

                    image = cv2.resize(img_8, (w,h), interpolation=cv2.INTER_CUBIC).transpose((2,0,1))
                    CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                    CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                    CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                    CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                    #MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                    loss_dict = {'loss':loss.item(),
                                 'loss_rvmin':loss_rvmin.item(),
                                 'loss_er':loss_er.item(),
                                 'loss_ecr':loss_ecr.item(),
                                 'bg_loss':bg_loss.item(),
                                 'fg_loss': fg_loss.item(),
                                 'neg_loss': neg_loss.item(),
                                 'loss_cls': loss_cls if type(loss_cls) == int else loss_cls.item()
                                 }
                    itr = optimizer.global_step - 1
                    tblogger.add_scalars('loss', loss_dict, itr)
                    tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                    tblogger.add_image('Image', input_img, itr)
                    #tblogger.add_image('Mask', MASK, itr)
                    tblogger.add_image('CLS1', CLS1, itr)
                    tblogger.add_image('CLS2', CLS2, itr)
                    tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                    tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                    tblogger.add_images('CAM1', CAM1, itr)
                    tblogger.add_images('CAM2', CAM2, itr)
                    tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                    tblogger.add_images('CAM_RV2', CAM_RV2, itr)

        else:
            print('')
            timer.reset_stage()

        torch.save(model.module.state_dict(), args.session_name + '.pth')

         
