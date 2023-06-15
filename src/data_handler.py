import sys, os
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import random
import csv
import pickle
import statsmodels.api as sm
from datetime import datetime
import scipy.stats as stats
from skimage.transform import resize

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms


def find_all_files(folder, suffix='npz'):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

def get_all_pids(data_dir):
    pids = []
    dict_pid_fid = {}
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        pid = raw_data['pid'].item()
        pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid

def get_all_pids_filter(data_dir, keep_list=None):
    race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

    pids = []
    dict_pid_fid = {}
    files = []
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        # race = int(raw_data['race'].item())
        race = raw_data['race'].item()
        if keep_list is not None and race not in keep_list:
            continue

        if not hasattr(raw_data, 'pid'):
            pid = f[f.find('_')+1:f.find('.')]
        else:
            pid = raw_data['pid'].item()
            pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
        files.append(f)
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid, files

class EyeFair(Dataset):
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0,
                    depth=1, indices=None, attribute_type='race', transform=None):

        self.data_path = data_path
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform

        self.data_files = find_all_files(self.data_path, suffix='npz')
        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]

        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        for x in self.data_files:
            rnflt_file = os.path.join(self.data_path, x)
            raw_data = np.load(rnflt_file, allow_pickle=True)
            min_vals.append(raw_data['md'].astype(np.float32).item())
            max_vals.append(raw_data['md'].astype(np.float32).item())
        print(f'min: {min(min_vals):.4f}, max: {max(max_vals):.4f}')
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rnflt_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(rnflt_file, allow_pickle=True)

        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        elif 'bscan' in self.modality_type:
            oct_img = raw_data['oct_bscans']
            oct_img_array = []
            for img in oct_img:
                oct_img_array.append(resize(img, (200, 200)))
            oct_img_array = np.stack(oct_img_array, axis=0)
            data_sample = np.stack([oct_img_array]*(1), axis=0).astype(float)
            if self.transform:
                data_sample = self.transform(data_sample)
        elif self.modality_type == 'ilm':
            ilm_sample = raw_data[self.modality_type]
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            data_sample = ilm_sample.astype(np.float32)
        elif self.modality_type == 'rnflt+ilm':
            rnflt_sample = raw_data['rnflt']
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            
            ilm_sample = raw_data['ilm']
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)

            data_sample = np.concatenate((rnflt_sample, ilm_sample), axis=0)
            data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'clockhours':
            data_sample = raw_data[self.modality_type].astype(np.float32)

        if self.task == 'md':
            y = torch.tensor(float(raw_data['md'].item()))
        elif self.task == 'tds':
            y = torch.tensor(float(raw_data['glaucoma'].item()))
        elif self.task == 'cls':
            y = torch.tensor(float(raw_data['glaucoma'].item()))

        attr = 0
        if self.attribute_type == 'race':
            cur_race = raw_data['race'].item()
            if cur_race in self.race_mapping:
                attr = self.race_mapping[cur_race]
            attr = torch.tensor(attr).int()
        elif self.attribute_type == 'gender':
            attr = torch.tensor(raw_data['male'].item()).int()

        return data_sample, y, attr

def load_data_(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, isHAVO=0, subset='train'
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not isHAVO:
        if not data_dir:
            raise ValueError("unspecified data directory")
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    elif isHAVO == 1:
        dataset = HAVO(data_dir, subset=subset, resolution=image_size)
    elif isHAVO == 2:
        dataset = HAVO_RNFLT(data_dir, subset=subset, resolution=image_size)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict
