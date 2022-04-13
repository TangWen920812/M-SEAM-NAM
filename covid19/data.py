import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from tool import imutils
from torchvision import transforms
import pydicom as dicom
import os
import random
import SimpleITK as sitk
import scipy.ndimage


IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']

CAT_NAME_TO_NUM = dict(zip(CAT_LIST,range(len(CAT_LIST))))

def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    el_list = minidom.parse(os.path.join(voc12_root, ANNOT_FOLDER_NAME,img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab

def load_image_label_list_from_xml(img_name_list, voc12_root):

    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]

def load_image_label_list_from_npy(img_name_list):

    cls_labels_dict = np.load('voc12/cls_labels.npy', allow_pickle=True).item()

    return [cls_labels_dict[img_name] for img_name in img_name_list]

def get_img_path(img_name, voc12_root):
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')

def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list]

    #img_name_list = img_gt_name_list
    return img_name_list

class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        img = PIL.Image.open(get_img_path(name, self.voc12_root)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return name, img


class VOC12ClsDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root, transform=None):
        super().__init__(img_name_list_path, voc12_root, transform)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)
        #self.label_list = load_image_label_list_from_xml(self.img_name_list, self.voc12_root)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label = torch.from_numpy(self.label_list[idx])

        return name, img, label

def load_txt(file):
    with open(file) as readfile:
        rowlist = readfile.readlines()
    path_list, cls_list = [], []
    for row in rowlist:
        path = row.split('\t')[0]
        cls = row.split('\t')[1]
        path_list.append(path.strip())
        cls_list.append(eval(cls.strip()))
    return path_list, cls_list

def get_dicom_imgarray(path):
    if os.path.exists(path):
        dcm = dicom.read_file(path)
        img = dcm.pixel_array
        img = (img - img.min()) / (img.max() - img.min()) * 255
    _path = path.split('/')
    tmp = _path[-1].split('_')
    tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) - 1))[-4:] + '.dcm'
    _path[-1] = '_'.join(tmp)
    path_up = '/'.join(_path)
    if os.path.exists(path_up):
        dcm = dicom.read_file(path_up)
        img_up = dcm.pixel_array
        img_up = (img_up - img_up.min()) / (img_up.max() - img_up.min()) * 255
    else:
        img_up = img
    _path = path.split('/')
    tmp = _path[-1].split('_')
    tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) + 1))[-4:] + '.dcm'
    _path[-1] = '_'.join(tmp)
    path_down = '/'.join(_path)
    if os.path.exists(path_down):
        dcm = dicom.read_file(path_down)
        img_down = dcm.pixel_array
        img_down = (img_down - img_down.min()) / (img_down.max() - img_down.min()) * 255
    else:
        img_down = img

    img_3c = np.array([img_up, img, img_down])
    # print(img_3c.shape)
    return img_3c.transpose((1, 2, 0))


class DicomDataset(Dataset):
    def __init__(self, dicom_root, txt_file, transform=None):
        self.img_name_list, self.cls_list = load_txt(txt_file)
        self.dicom_root = dicom_root
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def get_dicom_imgarray(self, path):
        if os.path.exists(path):
            dcm = dicom.read_file(path)
            img = dcm.pixel_array
            img = (img - img.min()) / (img.max() - img.min()) * 255
        _path = path.split('/')
        tmp = _path[-1].split('_')
        tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) - 1))[-4:] + '.dcm'
        _path[-1] = '_'.join(tmp)
        path_up = '/'.join(_path)
        if os.path.exists(path_up):
            dcm = dicom.read_file(path_up)
            img_up = dcm.pixel_array
            img_up = (img_up - img_up.min()) / (img_up.max() - img_up.min()) * 255
        else:
            img_up = img
        _path = path.split('/')
        tmp = _path[-1].split('_')
        tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) + 1))[-4:] + '.dcm'
        _path[-1] = '_'.join(tmp)
        path_down = '/'.join(_path)
        if os.path.exists(path_down):
            dcm = dicom.read_file(path_down)
            img_down = dcm.pixel_array
            img_down = (img_down - img_down.min()) / (img_down.max() - img_down.min()) * 255
        else:
            img_down = img

        img_3c = np.array([img_up, img, img_down])
        # print(img_3c.shape)
        return img_3c

    def get_other_item(self):
        try:
            idx = random.randint(0, len(self.img_name_list)-1)
            path = self.img_name_list[idx]
            name = path
            img = self.get_dicom_imgarray(os.path.join(self.dicom_root, path))
        except:
            idx = random.randint(0, len(self.img_name_list) - 1)
            path = self.img_name_list[idx]
            name = path
            img = self.get_dicom_imgarray(os.path.join(self.dicom_root, path))

        img = img.transpose((1, 2, 0))
        img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
        label = torch.from_numpy(np.array([1] if self.cls_list[idx][1]==1 else [-1]).astype('float32'))
        return name, img, label

    def __getitem__(self, idx):
        try:
            path = self.img_name_list[idx]
            name = path
            img = self.get_dicom_imgarray(os.path.join(self.dicom_root, path))
            img = img.transpose((1,2,0))
            img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
            label = torch.from_numpy(np.array([1] if self.cls_list[idx][1]==1 else [-1]).astype('float32'))
        except:
            name, img, label = self.get_other_item()

        if self.transform:
            img = self.transform(img)

        return name, img, label

class DicomDatasetMSF(DicomDataset):
    def __init__(self, dicom_root, txt_file, scales, inter_transform=None, unit=1):
        super().__init__(dicom_root, txt_file, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class DicomDatasetMI(Dataset):
    def __init__(self, dicom_root, txt_file, transform=None):
        self.pid_list, self.cls_list = load_txt(txt_file)
        self.dicom_root = dicom_root
        self.transform = transform
        self.bag_list, self.bag_cls = [], []
        pos_list, neg_list, unknow_list = [], [], []
        for i in range(len(self.pid_list)):
            if str(self.cls_list[i]) == str([0, 1]):
                pos_list.append(self.pid_list[i])
            elif str(self.cls_list[i]) == str([1, 0]):
                neg_list.append(self.pid_list[i])
            elif str(self.cls_list[i]) == str([0, 0]):
                unknow_list.append(self.pid_list[i])
        random.shuffle(pos_list)
        random.shuffle(neg_list)
        random.shuffle(unknow_list)

        for i in range(len(pos_list)):
            self.bag_list.append([pos_list[i]] + unknow_list[i*7:i*7+7])
            self.bag_cls.append([0, 1])
        for i in range(len(neg_list)//8):
            self.bag_list.append(neg_list[i*8:i*8+8])
            self.bag_cls.append([1, 0])

    def __len__(self):
        return len(self.bag_list)

    def get_dicom_imgarray(self, path):
        if os.path.exists(path):
            dcm = dicom.read_file(path, force=True)
            img = dcm.pixel_array
            img = (img - img.min()) / (img.max() - img.min()) * 255
        _path = path.split('/')
        tmp = _path[-1].split('_')
        tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) - 1))[-4:] + '.dcm'
        _path[-1] = '_'.join(tmp)
        path_up = '/'.join(_path)
        if os.path.exists(path_up):
            dcm = dicom.read_file(path_up)
            img_up = dcm.pixel_array
            img_up = (img_up - img_up.min()) / (img_up.max() - img_up.min()) * 255
        else:
            img_up = img
        _path = path.split('/')
        tmp = _path[-1].split('_')
        tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) + 1))[-4:] + '.dcm'
        _path[-1] = '_'.join(tmp)
        path_down = '/'.join(_path)
        if os.path.exists(path_down):
            dcm = dicom.read_file(path_down)
            img_down = dcm.pixel_array
            img_down = (img_down - img_down.min()) / (img_down.max() - img_down.min()) * 255
        else:
            img_down = img

        img_3c = np.array([img_up, img, img_down])
        # print(img_3c.shape)
        return img_3c

    def get_bag_img(self, idx):
        bag = self.bag_list[idx]
        # todo: shuffle
        random.shuffle(bag)
        img_list = []
        for pid in bag:
            img = self.get_dicom_imgarray(os.path.join(self.dicom_root, pid))
            img = img.transpose((1, 2, 0))
            img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')
            if self.transform:
                img = self.transform(img)
            img_list.append(img)

        return np.array(img_list)

    def __getitem__(self, idx):
        pid_bag = self.bag_list[idx]
        bag = self.get_bag_img(idx)
        label = self.bag_cls[idx]
        label = torch.from_numpy(np.array([1] if label[1] == 1 else [-1]).astype('float32'))

        return pid_bag, torch.from_numpy(bag), label

class DicomUnetDataset(Dataset):
    def __init__(self, dicom_root, txt_file, transform=None, transform_l=None):
        self.img_name_list, self.cls_list = load_txt(txt_file)
        self.dicom_root = dicom_root
        self.transform = transform
        self.transform_l = transform_l

    def __len__(self):
        return len(self.img_name_list)

    def get_dicom_imgarray(self, path):
        if os.path.exists(path):
            dcm = dicom.read_file(path)
            img = dcm.pixel_array
            img = (img - img.min()) / (img.max() - img.min()) * 255
        _path = path.split('/')
        tmp = _path[-1].split('_')
        tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) - 1))[-4:] + '.dcm'
        _path[-1] = '_'.join(tmp)
        path_up = '/'.join(_path)
        if os.path.exists(path_up):
            dcm = dicom.read_file(path_up)
            img_up = dcm.pixel_array
            img_up = (img_up - img_up.min()) / (img_up.max() - img_up.min()) * 255
        else:
            img_up = img
        _path = path.split('/')
        tmp = _path[-1].split('_')
        tmp[-1] = ('0000' + str(int(_path[-1].split('_')[-1][:-4]) + 1))[-4:] + '.dcm'
        _path[-1] = '_'.join(tmp)
        path_down = '/'.join(_path)
        if os.path.exists(path_down):
            dcm = dicom.read_file(path_down)
            img_down = dcm.pixel_array
            img_down = (img_down - img_down.min()) / (img_down.max() - img_down.min()) * 255
        else:
            img_down = img

        img_3c = np.array([img_up, img, img_down])
        # print(img_3c.shape)
        return img_3c

    def __getitem__(self, idx):
        path = self.img_name_list[idx]
        name = path
        img = self.get_dicom_imgarray(os.path.join(self.dicom_root, path))
        img = img.transpose((1,2,0))
        img = PIL.Image.fromarray(img.astype('uint8')).convert('RGB')

        tmp = True if self.cls_list[idx][1]==1 else False
        if tmp:
            label = sitk.ReadImage(os.path.join(self.dicom_root,
                                                path.replace('data', 'anno').replace('dcm', 'nrrd')))
            label = sitk.GetArrayFromImage(label)
            label = torch.from_numpy(label.T.astype('float'))
        else:
            label = torch.from_numpy(np.zeros((512, 512)).astype('float'))

        if self.transform:
            img = self.transform(img)
        if self.transform_l:
            # label = scipy.ndimage.zoom(label, 448/512, order=0)
            label = self.transform_l(label)

        return name, img, label

class VOC12ClsDatasetMSF(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        msf_img_list = []
        for i in range(len(ms_img_list)):
            msf_img_list.append(ms_img_list[i])
            msf_img_list.append(np.flip(ms_img_list[i], -1).copy())

        return name, msf_img_list, label

class VOC12ClsDatasetMS(VOC12ClsDataset):

    def __init__(self, img_name_list_path, voc12_root, scales, inter_transform=None, unit=1):
        super().__init__(img_name_list_path, voc12_root, transform=None)
        self.scales = scales
        self.unit = unit
        self.inter_transform = inter_transform

    def __getitem__(self, idx):
        name, img, label = super().__getitem__(idx)

        rounded_size = (int(round(img.size[0]/self.unit)*self.unit), int(round(img.size[1]/self.unit)*self.unit))

        ms_img_list = []
        for s in self.scales:
            target_size = (round(rounded_size[0]*s),
                           round(rounded_size[1]*s))
            s_img = img.resize(target_size, resample=PIL.Image.CUBIC)
            ms_img_list.append(s_img)

        if self.inter_transform:
            for i in range(len(ms_img_list)):
                ms_img_list[i] = self.inter_transform(ms_img_list[i])

        return name, ms_img_list, label

class ExtractAffinityLabelInRadius():

    def __init__(self, cropsize, radius=5):
        self.radius = radius

        self.search_dist = []

        for x in range(1, radius):
            self.search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius+1, radius):
                if x*x + y*y < radius*radius:
                    self.search_dist.append((y, x))

        self.radius_floor = radius-1

        self.crop_height = cropsize - self.radius_floor
        self.crop_width = cropsize - 2 * self.radius_floor
        return

    def __call__(self, label):

        labels_from = label[:-self.radius_floor, self.radius_floor:-self.radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in self.search_dist:
            labels_to = label[dy:dy+self.crop_height, self.radius_floor+dx:self.radius_floor+dx+self.crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), torch.from_numpy(neg_affinity_label)

class VOC12AffDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_la_dir, label_ha_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_la_dir = label_la_dir
        self.label_ha_dir = label_ha_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_la_path = os.path.join(self.label_la_dir, name + '.npy')

        label_ha_path = os.path.join(self.label_ha_dir, name + '.npy')

        label_la = np.load(label_la_path, allow_pickle=True).item()
        label_ha = np.load(label_ha_path, allow_pickle=True).item()

        label = np.array(list(label_la.values()) + list(label_ha.values()))
        label = np.transpose(label, (1, 2, 0))

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        no_score_region = np.max(label, -1) < 1e-5
        label_la, label_ha = np.array_split(label, 2, axis=-1)
        label_la = np.argmax(label_la, axis=-1).astype(np.uint8)
        label_ha = np.argmax(label_ha, axis=-1).astype(np.uint8)
        label = label_la.copy()
        label[label_la == 0] = 255
        label[label_ha == 0] = 0
        label[no_score_region] = 255 # mostly outer of cropped region
        label = self.extract_aff_lab_func(label)

        return img, label

class VOC12AffGtDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, label_dir, cropsize, voc12_root, radius=5,
                 joint_transform_list=None, img_transform_list=None, label_transform_list=None):
        super().__init__(img_name_list_path, voc12_root, transform=None)

        self.label_dir = label_dir
        self.voc12_root = voc12_root

        self.joint_transform_list = joint_transform_list
        self.img_transform_list = img_transform_list
        self.label_transform_list = label_transform_list

        self.extract_aff_lab_func = ExtractAffinityLabelInRadius(cropsize=cropsize//8, radius=radius)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name, img = super().__getitem__(idx)

        label_path = os.path.join(self.label_dir, name + '.png')

        label = scipy.misc.imread(label_path)

        for joint_transform, img_transform, label_transform \
                in zip(self.joint_transform_list, self.img_transform_list, self.label_transform_list):

            if joint_transform:
                img_label = np.concatenate((img, label), axis=-1)
                img_label = joint_transform(img_label)
                img = img_label[..., :3]
                label = img_label[..., 3:]

            if img_transform:
                img = img_transform(img)
            if label_transform:
                label = label_transform(label)

        label = self.extract_aff_lab_func(label)

        return img, label
