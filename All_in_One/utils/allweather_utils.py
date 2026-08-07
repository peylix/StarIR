"""Datasets for the AllWeather setting (Outdoor-Rain + RainDrop + Snow100K).

Training data layout (single mixed folder):
    <data_dir>/input/    degraded images
    <data_dir>/gt/       clean images (same filenames as input/)
    <data_dir>/gt_val/   optional: extra gt folder; its images are folded
                         into the training set (no validation split is used,
                         consistent with the StarIR paper)

Test set layouts (one folder per task):
    Outdoor-Rain:  <root>/data/  +  <root>/gt/     (im_0001_s80_a04.png -> im_0001.png)
    RainDrop:      <root>/data/  +  <root>/gt/     (0_rain.png -> 0_clean.png)
    Snow100K-L:    <root>/synthetic/  +  <root>/gt/  (same filenames)
"""
import os
import random
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import random_augmentation

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def list_images(folder):
    names = [n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTENSIONS)]
    names.sort()
    return names


class GtIndex:
    """Index of a gt folder that resolves degraded filenames to gt filenames."""

    def __init__(self, gt_dir):
        self.gt_dir = gt_dir
        self.names = set(list_images(gt_dir))
        self.by_stem = {}
        for name in sorted(self.names):
            self.by_stem.setdefault(os.path.splitext(name)[0], name)

    def resolve(self, input_name):
        """Return the gt filename for a degraded filename, or None."""
        if input_name in self.names:
            return input_name
        stem = os.path.splitext(input_name)[0]
        candidates = [stem]
        # RainDrop naming: '0_rain' -> '0_clean'
        if stem.endswith('_rain'):
            candidates.append(stem[:-len('_rain')] + '_clean')
        # Outdoor-Rain naming: 'im_0001_s80_a04' -> 'im_0001'
        stripped = re.sub(r'_s\d+(\.\d+)?_a\d+(\.\d+)?$', '', stem)
        if stripped != stem:
            candidates.append(stripped)
        for cand in candidates:
            if cand in self.by_stem:
                return self.by_stem[cand]
        return None


def build_pairs(input_dir, gt_dirs, verbose=True):
    """Match every image in input_dir to its gt, trying gt_dirs in order.

    Returns a list of (input_path, gt_path).
    """
    if isinstance(gt_dirs, str):
        gt_dirs = [gt_dirs]
    indexes = [GtIndex(d) for d in gt_dirs if os.path.isdir(d)]
    pairs, unmatched = [], []
    for name in list_images(input_dir):
        for index in indexes:
            gt_name = index.resolve(name)
            if gt_name is not None:
                pairs.append((os.path.join(input_dir, name),
                              os.path.join(index.gt_dir, gt_name)))
                break
        else:
            unmatched.append(name)
    if verbose:
        print('[AllWeather] %s: %d pairs, %d unmatched'
              % (input_dir, len(pairs), len(unmatched)))
        if unmatched:
            print('[AllWeather] unmatched examples: %s' % ', '.join(unmatched[:5]))
    return pairs


def _load_rgb(path):
    return np.array(Image.open(path).convert('RGB'))


def _align_sizes(img_a, img_b):
    """Crop two images to their common top-left region if sizes differ."""
    h = min(img_a.shape[0], img_b.shape[0])
    w = min(img_a.shape[1], img_b.shape[1])
    return img_a[:h, :w], img_b[:h, :w]


class AllWeatherTrainDataset(Dataset):
    """Random patches from the mixed AllWeather training set."""

    def __init__(self, data_dir, patch_size=128, input_subdir='input',
                 gt_subdir='gt', extra_gt_subdir='gt_val'):
        super(AllWeatherTrainDataset, self).__init__()
        input_dir = os.path.join(data_dir, input_subdir)
        # gt_val is a secondary gt source: it only matters for inputs whose gt
        # was moved out of gt/, so the full set is always trained on
        gt_dirs = [os.path.join(data_dir, gt_subdir),
                   os.path.join(data_dir, extra_gt_subdir)]
        self.pairs = build_pairs(input_dir, gt_dirs)
        if not self.pairs:
            raise RuntimeError('No training pairs found under %s' % data_dir)
        self.patch_size = patch_size
        self.toTensor = ToTensor()

    def _crop_patch(self, img_1, img_2):
        ps = self.patch_size
        pad_h = max(0, ps - img_1.shape[0])
        pad_w = max(0, ps - img_1.shape[1])
        if pad_h or pad_w:
            mode = 'reflect' if (pad_h < img_1.shape[0] and pad_w < img_1.shape[1]) else 'edge'
            img_1 = np.pad(img_1, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
            img_2 = np.pad(img_2, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        ind_h = random.randint(0, img_1.shape[0] - ps)
        ind_w = random.randint(0, img_1.shape[1] - ps)
        return (img_1[ind_h:ind_h + ps, ind_w:ind_w + ps],
                img_2[ind_h:ind_h + ps, ind_w:ind_w + ps])

    def __getitem__(self, idx):
        input_path, gt_path = self.pairs[idx]
        degrad_img = _load_rgb(input_path)
        clean_img = _load_rgb(gt_path)
        degrad_img, clean_img = _align_sizes(degrad_img, clean_img)

        degrad_patch, clean_patch = random_augmentation(
            *self._crop_patch(degrad_img, clean_img))

        name = os.path.splitext(os.path.basename(input_path))[0]
        return [name], self.toTensor(degrad_patch), self.toTensor(clean_patch)

    def __len__(self):
        return len(self.pairs)


class AllWeatherTestDataset(Dataset):
    """A benchmark test set: full degraded images plus the path of their gt."""

    def __init__(self, input_dir, gt_dir):
        super(AllWeatherTestDataset, self).__init__()
        index = GtIndex(gt_dir)
        self.items = []
        missing = []
        for name in list_images(input_dir):
            gt_name = index.resolve(name)
            if gt_name is None:
                missing.append(name)
            else:
                self.items.append((os.path.join(input_dir, name),
                                   os.path.join(gt_dir, gt_name)))
        if missing:
            print('[AllWeather] %d inputs in %s have no gt match, e.g.: %s'
                  % (len(missing), input_dir, ', '.join(missing[:5])))
        if not self.items:
            raise RuntimeError('No test pairs found for %s' % input_dir)
        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        input_path, gt_path = self.items[idx]
        degraded_img = _load_rgb(input_path)
        name = os.path.splitext(os.path.basename(input_path))[0]
        return [name], self.toTensor(degraded_img), gt_path

    def __len__(self):
        return len(self.items)
