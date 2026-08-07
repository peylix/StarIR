"""PSNR / SSIM / MAE / LPIPS / DISTS evaluation.

Ported from the reference metrics package (SwinIR-style Y-channel PSNR/SSIM,
LPIPS with AlexNet backbone, DISTS from piq) so results are directly
comparable with numbers produced by that code.
"""
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm


def _convert_input_type_range(img):
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)


def bgr2ycbcr(img, y_only=False):
    """ITU-R BT.601 conversion (matches the MATLAB/SwinIR convention)."""
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img


def to_y_channel(img):
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def calculate_psnr(img1, img2, test_y_channel=True):
    """PSNR on images with range [0, 255] (BGR, HWC)."""
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, test_y_channel=True):
    """SSIM on images with range [0, 255] (BGR, HWC)."""
    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    assert img1.shape[2] == 3
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calculate_mae(img1, img2, test_y_channel=True):
    """Mean absolute error on [0, 1] scale."""
    assert img1.shape == img2.shape
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
    return float(np.mean(np.abs(img1 - img2)) / 255.0)


def _bgr_to_rgb_tensor(img_bgr, device):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return tensor


class PerceptualMetricComputer:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._lpips_model = None
        self._dists_model = None

    def _get_lpips_model(self):
        if self._lpips_model is None:
            try:
                import lpips
            except ImportError as exc:
                raise ImportError('LPIPS requires `pip install lpips`.') from exc
            self._lpips_model = lpips.LPIPS(net='alex').to(self.device).eval()
        return self._lpips_model

    def _get_dists_model(self):
        if self._dists_model is None:
            try:
                from piq import DISTS
            except ImportError as exc:
                raise ImportError('DISTS requires `pip install piq`.') from exc
            self._dists_model = DISTS().to(self.device).eval()
        return self._dists_model

    @torch.no_grad()
    def calculate_lpips(self, img1, img2):
        model = self._get_lpips_model()
        pred = _bgr_to_rgb_tensor(img1, self.device) * 2.0 - 1.0
        gt = _bgr_to_rgb_tensor(img2, self.device) * 2.0 - 1.0
        return float(model(pred, gt).item())

    @torch.no_grad()
    def calculate_dists(self, img1, img2):
        model = self._get_dists_model()
        pred = _bgr_to_rgb_tensor(img1, self.device)
        gt = _bgr_to_rgb_tensor(img2, self.device)
        return float(model(pred, gt).item())


def summarize_metric(values):
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan'), float('nan')
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def format_mean_std(mean, std, decimals=4):
    return f'{mean:.{decimals}f} ± {std:.{decimals}f}'


METRIC_KEYS = ['psnr', 'ssim', 'mae', 'lpips', 'dists']


def print_metric_summary(summary, decimals=4, title=None):
    lines = []
    if title:
        lines.append(title)
    for key in METRIC_KEYS:
        lines.append('%s: %s' % (key.upper().ljust(5), format_mean_std(
            summary[f'{key}_mean'], summary[f'{key}_std'], decimals=decimals)))
    text = '\n'.join(lines)
    print(text)
    return text


def _load_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return img


def _center_crop_to_common(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    def crop(img):
        top = (img.shape[0] - h) // 2
        left = (img.shape[1] - w) // 2
        return img[top:top + h, left:left + w]
    return crop(img1), crop(img2)


def evaluate_pairs(pairs, device=None, test_y_channel=True, verbose=False, progress=True):
    """Evaluate a list of (pred_path, gt_path) pairs.

    Returns a dict with <metric>_mean / <metric>_std / <metric> (per-image list).
    """
    metric_computer = PerceptualMetricComputer(device=device)
    values = {key: [] for key in METRIC_KEYS}

    iterator = tqdm(pairs, desc='evaluating', ncols=80) if progress else pairs
    for pred_path, gt_path in iterator:
        pred = _load_bgr(pred_path)
        gt = _load_bgr(gt_path)
        if pred.shape != gt.shape:
            print('Shape mismatch %s %s vs %s %s -> center-cropping to common size'
                  % (os.path.basename(pred_path), pred.shape,
                     os.path.basename(gt_path), gt.shape))
            pred, gt = _center_crop_to_common(pred, gt)

        values['psnr'].append(calculate_psnr(pred, gt, test_y_channel=test_y_channel))
        values['ssim'].append(calculate_ssim(pred, gt, test_y_channel=test_y_channel))
        values['mae'].append(calculate_mae(pred, gt, test_y_channel=test_y_channel))
        values['lpips'].append(metric_computer.calculate_lpips(pred, gt))
        values['dists'].append(metric_computer.calculate_dists(pred, gt))

        if verbose:
            print('%s: PSNR=%.4f, SSIM=%.4f, MAE=%.4f, LPIPS=%.4f, DISTS=%.4f'
                  % (os.path.basename(pred_path), values['psnr'][-1],
                     values['ssim'][-1], values['mae'][-1],
                     values['lpips'][-1], values['dists'][-1]))

    summary = {}
    for key in METRIC_KEYS:
        mean, std = summarize_metric(values[key])
        summary[f'{key}_mean'] = mean
        summary[f'{key}_std'] = std
        summary[key] = values[key]
    summary['num_images'] = len(pairs)
    return summary
