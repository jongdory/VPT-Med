import json
import cv2
import os
import numpy as np
import random
from PIL import Image
from monai import transforms
from monai.data import Dataset as MonaiDataset
from torch.utils.data import Dataset
from einops import rearrange, repeat
from scipy.ndimage import gaussian_filter
import open_clip
import torch
from scipy.ndimage import center_of_mass, distance_transform_edt

class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        norm_img =  cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        canny = cv2.Canny(norm_img, low_threshold, high_threshold)
        return cv2.normalize(canny, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

brats_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["T1", "T1ce", "T2", "FLAIR", "seg"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["T1", "T1ce", "T2", "FLAIR", "seg"], allow_missing_keys=True),
        transforms.Orientationd(keys=["T1", "T1ce", "T2", "FLAIR", "seg"], axcodes="RAI", allow_missing_keys=True),
        transforms.SpatialPadd(keys=["T1", "T1ce", "T2", "FLAIR", "seg"], spatial_size=(256, 256, 155), allow_missing_keys=True),
        transforms.Lambdad(keys=["T1", "T1ce", "T2", "FLAIR", "seg"], func=lambda x: x[0, :, :, :], allow_missing_keys=True),
        transforms.EnsureTyped(keys=["T1", "T1ce", "T2", "FLAIR", "seg"], allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(keys=["T1", "T1ce", "T2", "FLAIR"], lower=0, upper=100, b_min=0, b_max=1, allow_missing_keys=True),
        transforms.Lambdad(keys=["T1", "T1ce", "T2", "FLAIR"], func=lambda x: np.where(x > 1, 1, x), allow_missing_keys=True),
    ]
)

amos_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["CT"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["CT"], allow_missing_keys=True),
        transforms.Orientationd(keys=["CT"], axcodes="RPI", allow_missing_keys=True),
        transforms.Lambdad(keys=["CT"], func=lambda x: x[0, :, :, :]),
        transforms.EnsureTyped(keys=["CT"]),
        transforms.ScaleIntensityRanged(keys=["CT"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
    ]
)

def get_brats_dataset(data_path):
    transform = brats_transforms
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz") 
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz") 
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz") 
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz") 
        seg = os.path.join(sub_path, f"{subject}_seg.nii.gz")

        data.append({"T1":t1, "T1ce":t1ce, "T2":t2, "FLAIR":flair, "seg":seg, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

def get_amos_dataset(data_path):
    transform = amos_transforms

    data = []
    phases = ["imagesTr", "imagesVa"]
    for phase in phases:
        phase_path = os.path.join(data_path, phase)
        for subject in os.listdir(phase_path):
            if not subject.startswith("amos"): continue
            ct = os.path.join(phase_path, subject)
            if os.path.exists(ct):
                data.append({"CT":ct, "subject_id": subject.split(".")[0]})

    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


def add_rician_noise(image, noise_level=1):
    gaussian_noise_1 = np.random.normal(0, noise_level, image.shape)
    gaussian_noise_2 = np.random.normal(0, noise_level, image.shape)

    noisy_image = np.sqrt((image + gaussian_noise_1)**2 + gaussian_noise_2**2)

    noisy_image = np.clip(noisy_image, 0, 1) 
    noisy_image = noisy_image.astype(np.float32)

    return noisy_image

def add_mixed_poisson_gaussian_noise(image, poisson_scale=1, gaussian_sigma=1):
    noisy_image = np.random.poisson(image * poisson_scale) / poisson_scale
    noisy_image = gaussian_filter(noisy_image, sigma=gaussian_sigma)

    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image = noisy_image.astype(np.float32)

    return noisy_image

class MyDataset(Dataset):
    def __init__(self, brats_datapath, amos_datapath):
        self.brats_data = get_brats_dataset(brats_datapath)
        self.amos_data = get_amos_dataset(amos_datapath)
        self.canny = CannyDetector()

    def __len__(self):
        return len(self.brats_data) + len(self.amos_data)
    
    def tumor_crop(self, x, x_seg):
        x_seg = cv2.dilate(x_seg, np.ones((5, 5), np.uint8), iterations=1)

        x[x_seg == 1] = 1

        return x
    
    def normal_crop(self, x, x_seg):
        tumor_mask = (x_seg == 1)
        brain_mask = (x > 0)

        if x_seg.sum() == 0:
            nonzero_indices = np.transpose(np.nonzero(brain_mask))
            if nonzero_indices.shape[0] == 0:
                return x
            random_index = np.random.choice(nonzero_indices.shape[0])
            cy, cx = nonzero_indices[random_index]
        
            radius = np.random.randint(10, 30)
            mask = np.zeros_like(x_seg)

            Y, X = np.ogrid[:x_seg.shape[0], :x_seg.shape[1]]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask[dist <= radius] = 1
        else:
            distance_map = distance_transform_edt(np.logical_and(brain_mask, ~tumor_mask))
            cx, cy = np.unravel_index(distance_map.argmax(), distance_map.shape)
            seg_cx, seg_cy = center_of_mass(x_seg)
            dx, dy = int(cx - seg_cx), int(cy - seg_cy)

            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            mask = cv2.warpAffine(x_seg, translation_matrix, (x_seg.shape[1], x_seg.shape[0]))
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        x[mask == 1] = 1
        
        return x

    def __getitem__(self, idx):
        prob = np.random.rand()

        # BRATS
        if idx < len(self.brats_data):
            modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
            item = self.brats_data[idx]

            src = np.random.choice(modalities)
            modalities.remove(src)
            tgt = np.random.choice(modalities)

            source = rearrange(item[src].numpy().copy(), "h w d -> w h d")
            source = source[::-1, ::-1, ::-1]

            # Find the slice without background.
            indices = np.where(source.sum(axis=(0, 1)) > 0)[0]
            start_idx, end_idx = indices[0], indices[-1]
            # Randomly select a slice.
            slice_idx = np.random.randint(start_idx+20, end_idx-20)
            slice_num = source.shape[2]

            canny = None

            # Modality Translation (25%)
            if prob < 0.25:
                target = rearrange(item[tgt].numpy().copy(), "h w d -> w h d")
                target = target[::-1, ::-1, ::-1]

                source = source[:,:,slice_idx]
                target = target[:,:,slice_idx]

                source = repeat(source, "h w -> h w c", c=3)
                target = repeat(target, "h w -> h w c", c=3)

                prompt = f"Brain MR image,{tgt} modality,Translation, "

                canny = self.canny(source, 50, 150)
                canny = repeat(canny, "h w -> h w c", c=3)
                canny = cv2.resize(canny, (512, 512), interpolation=cv2.INTER_CUBIC)
            
            # Super Resolution (25%)
            elif prob < 0.5:
                source = source[:,:,slice_idx]
                target = source.copy()

                source = repeat(source, "h w -> h w c", c=3)
                target = repeat(target, "h w -> h w c", c=3)

                # interpolate for source
                res = int(source.shape[0] / np.random.randint(3, 7))

                source = cv2.resize(source, (res, res), interpolation=cv2.INTER_AREA)

                prompt = f"Brain MR image,{src} modality,Super-Resolution, "
            
            # Denoising (25%)
            elif prob < 0.75:
                source = source[:,:,slice_idx]
                target = source.copy()

                # add noise
                source = add_rician_noise(source, noise_level=np.random.randint(400, 700)/10000)

                source = repeat(source, "h w -> h w c", c=3)
                target = repeat(target, "h w -> h w c", c=3)

                prompt = f"Brain MR image,{src} modality,Denoising, "
            # Inpainting (25%)
            else:
                seg = rearrange(item['seg'].numpy().copy(), "h w d -> w h d")
                seg = seg[::-1, ::-1, ::-1]
                seg[seg == 4] = 1
                seg[seg == 2] = 0

                # Tumor (12.5%)
                if prob < 0.875:
                    tumor_indices = np.where(seg.sum(axis=(0, 1)) > 0)[0]
                    if len(tumor_indices) == 0:
                        prob = 1
                    else:
                        tumor_start_idx, tumor_end_idx = tumor_indices[0], tumor_indices[-1]
                        slice_idx = np.random.randint(tumor_start_idx+2, tumor_end_idx-2)

                source = source[:,:,slice_idx]
                target = source.copy()
                seg = seg[:,:,slice_idx]

                if prob < 0.9:
                    source = self.tumor_crop(source, seg)
                    region = "tumor"
                else:
                    source = self.normal_crop(source, seg)
                    region = "normal"

                source = repeat(source, "h w -> h w c", c=3)
                target = repeat(target, "h w -> h w c", c=3)

                prompt = f"Brain MR image,{src} modality,Inpainting, {region} image"

            source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_CUBIC)
            target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_CUBIC)

            if canny is None:
                canny = np.zeros_like(source)

        # AMOS
        elif idx < len(self.brats_data) + len(self.amos_data):
            item = self.amos_data[idx - len(self.brats_data)]
            ct = item['CT']

            ct = rearrange(ct, "h w d -> w h d")

            source = ct.numpy().copy()[:, ::-1]

            canny = None

            # Randomly select a slice.
            slice_idx = np.random.randint(0, ct.shape[2])

            # Super Resolution (50%)
            if prob < 0.5:
                source = source[:,:,slice_idx]
                target = source.copy()
                
                source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)

                # interpolate for source
                res = int(512 / np.random.randint(3, 7))
                source = cv2.resize(source, (res, res), interpolation=cv2.INTER_AREA)
                source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_CUBIC)
                target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

                prompt = f"Abdominal CT,CT modality,Super-Resolution, "

            # Denoising (50%)
            else:
                source = source[:,:,slice_idx]
                target = source.copy()

                # add noise
                source = add_mixed_poisson_gaussian_noise(source, poisson_scale=random.uniform(8, 12), gaussian_sigma=random.uniform(0.5, 1.5))

                source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
                target = cv2.cvtColor(target, cv2.COLOR_GRAY2RGB)

                source = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)
                target = cv2.resize(target, (512, 512), interpolation=cv2.INTER_AREA)

                prompt = f"Abdominal CT,CT modality,Denoising, "

            if canny is None:
                canny = np.zeros_like(source)

        source = np.clip(source, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

        target = target * 2 -1
        d_prompt = ""

        return dict(jpg=target, txt=prompt, dtxt=d_prompt, hint=canny, source=source)
    

def get_brats_testset(data_path):
    transform = brats_transforms
    
    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        if os.path.exists(sub_path) == False: continue
        t1 = os.path.join(sub_path, f"{subject}_t1.nii.gz") 
        t1ce = os.path.join(sub_path, f"{subject}_t1ce.nii.gz") 
        t2 = os.path.join(sub_path, f"{subject}_t2.nii.gz") 
        flair = os.path.join(sub_path, f"{subject}_flair.nii.gz") 

        data.append({"T1":t1, "T1ce":t1ce, "T2":t2, "FLAIR":flair, "subject_id": subject})
                    
    print("num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

class MyTestset(Dataset):
    def __init__(self, data_path, is_canny=False, src="T1", tgt="T2"):
        self.brats_data = get_brats_testset(data_path)
        self.canny = CannyDetector()
        self.is_canny = is_canny
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.brats_data)

    def __getitem__(self, idx):
        source = self.brats_data[idx][self.src].numpy().copy()
        subject_id = self.brats_data[idx]["subject_id"]
        source = rearrange(source, "h w d -> w h d")
        source = source[::-1, ::-1, ::-1]
        h, w, d = source.shape
        source = repeat(source, "w h d -> c w h d", c=3)

        canny = None
        if self.is_canny:
            canny = np.zeros((h, w, 3, d))
            for i in range(d):
                canny_d = self.canny(source[:,:,i], 50, 150)
                canny_d = repeat(canny_d, "w h -> w h c", c=3)
            canny = rearrange(canny, "w h c d -> c w h d")
        else:
            canny = np.zeros_like(source)

        d_prompt = ""

        return dict(dtxt=d_prompt, hint=canny, source=source, subject_id=subject_id)