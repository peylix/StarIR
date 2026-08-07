"""Test StarIR on the three AllWeather benchmarks and report full metrics
(PSNR / SSIM / MAE / LPIPS / DISTS).

Example:
    python test_allweather.py --ckpt ckpt/allweather/last.ckpt \
        --outdoor_rain_dir ~/autodl-tmp/test/CVPR19RainTrain/test \
        --raindrop_dir     ~/autodl-tmp/test/raindrop_data/test_a \
        --snow100k_dir     ~/autodl-tmp/test/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L \
        --output_path results_allweather/
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from net.model import StarIR
from utils.allweather_utils import AllWeatherTestDataset
from utils.full_metrics import evaluate_pairs, print_metric_summary


def load_checkpoint(net, ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('state_dict', 'params_ema', 'params'):
            if key in ckpt:
                state = ckpt[key]
                break
    cleaned = {}
    for k, v in state.items():
        if k.startswith('net.'):
            k = k[len('net.'):]
        elif k.startswith('module.'):
            k = k[len('module.'):]
        cleaned[k] = v
    try:
        net.load_state_dict(cleaned, strict=True)
    except RuntimeError:
        missing, unexpected = net.load_state_dict(cleaned, strict=False)
        print('Loaded with strict=False. Missing keys: %s, unexpected keys: %s'
              % (missing, unexpected))
    return net


def save_restored(tensor, path):
    """Save a [1,C,H,W] tensor in [0,1] as PNG."""
    array = np.clip(tensor.squeeze(0).detach().cpu().numpy(), 0, 1)
    array = (array * 255.0).round().astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(array).save(path)


@torch.no_grad()
def run_inference(net, dataset, out_dir, device, num_workers=4, factor=32):
    """Restore all images, returning the per-image inference time in seconds."""
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False,
                        num_workers=num_workers)
    use_cuda = torch.cuda.is_available() and 'cuda' in str(device)
    times = []
    for ([name], degraded, gt_path) in tqdm(loader, desc='inference', ncols=80):
        degraded = degraded.to(device)
        if use_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        _, _, h, w = degraded.shape
        h_n = (factor - h % factor) % factor
        w_n = (factor - w % factor) % factor
        padded = F.pad(degraded, (0, w_n, 0, h_n), mode='reflect')
        restored = net(padded)[:, :, :h, :w]
        if use_cuda:
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        save_restored(restored, os.path.join(out_dir, name[0] + '.png'))
    return times


def run_task(net, task_name, root, input_subdir, gt_subdir, opt, device):
    input_dir = os.path.join(root, input_subdir)
    gt_dir = os.path.join(root, gt_subdir)
    if not os.path.isdir(input_dir) or not os.path.isdir(gt_dir):
        print('[%s] skipped: %s or %s not found' % (task_name, input_dir, gt_dir))
        return None

    print('\n===== %s =====' % task_name)
    dataset = AllWeatherTestDataset(input_dir, gt_dir)
    out_dir = os.path.join(opt.output_path, task_name)

    times = None
    if not opt.skip_inference:
        times = run_inference(net, dataset, out_dir, device,
                              num_workers=opt.num_workers)

    pairs = []
    for input_path, gt_path in dataset.items:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        pairs.append((os.path.join(out_dir, stem + '.png'), gt_path))

    summary = evaluate_pairs(pairs, device=device, verbose=opt.verbose)
    if times is not None and len(times) > 1:
        timed = times[1:]  # first image excluded (warmup/lazy initialization)
        summary['time_mean'] = float(np.mean(timed))
        summary['time_std'] = float(np.std(timed, ddof=1)) if len(timed) > 1 else 0.0

    text = print_metric_summary(summary, decimals=4,
                                title='%s (%d images, mean ± std):'
                                % (task_name, summary['num_images']))
    if 'time_mean' in summary:
        time_line = ('Time : %.4f ± %.4f s/image (first image excluded)'
                     % (summary['time_mean'], summary['time_std']))
        print(time_line)
        text += '\n' + time_line
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(text + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--outdoor_rain_dir', type=str, default='',
                        help='e.g. .../CVPR19RainTrain/test (contains data/, gt/)')
    parser.add_argument('--raindrop_dir', type=str, default='',
                        help='e.g. .../raindrop_data/test_a (contains data/, gt/)')
    parser.add_argument('--snow100k_dir', type=str, default='',
                        help='e.g. .../test/Snow100K-L (contains synthetic/, gt/)')
    parser.add_argument('--output_path', type=str, default='results_allweather/')
    parser.add_argument('--skip_inference', action='store_true',
                        help='only evaluate previously saved outputs')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--verbose', action='store_true', help='print per-image metrics')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)
    device = 'cuda:%d' % opt.cuda if torch.cuda.is_available() else 'cpu'

    net = StarIR().to(device)
    load_checkpoint(net, opt.ckpt)
    net.eval()
    print('Loaded checkpoint: %s' % opt.ckpt)

    num_params = sum(p.numel() for p in net.parameters())
    params_line = 'Parameters: %.2f M (%d)' % (num_params / 1e6, num_params)
    print(params_line)

    tasks = [
        ('outdoor_rain', opt.outdoor_rain_dir, 'data', 'gt'),
        ('raindrop', opt.raindrop_dir, 'data', 'gt'),
        ('snow100k_L', opt.snow100k_dir, 'synthetic', 'gt'),
    ]

    summaries = {}
    for task_name, root, input_subdir, gt_subdir in tasks:
        if not root:
            continue
        summary = run_task(net, task_name, os.path.expanduser(root),
                           input_subdir, gt_subdir, opt, device)
        if summary is not None:
            summaries[task_name] = summary

    if summaries:
        os.makedirs(opt.output_path, exist_ok=True)
        lines = [params_line, '',
                 'task\tN\tPSNR\tSSIM\tMAE\tLPIPS\tDISTS\tTime(s/image)']
        for task_name, s in summaries.items():
            cells = [task_name, '%d' % s['num_images']]
            for key in ('psnr', 'ssim', 'mae', 'lpips', 'dists'):
                cells.append('%.4f ± %.4f' % (s[key + '_mean'], s[key + '_std']))
            if 'time_mean' in s:
                cells.append('%.4f ± %.4f' % (s['time_mean'], s['time_std']))
            else:
                cells.append('n/a')
            lines.append('\t'.join(cells))
        table = '\n'.join(lines)
        print('\n===== Summary =====\n' + table)
        with open(os.path.join(opt.output_path, 'metrics_summary.txt'), 'w') as f:
            f.write(table + '\n')


if __name__ == '__main__':
    main()
