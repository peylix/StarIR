# StarIR on AllWeather (baseline setup)

Adaptation of StarIR's all-in-one pipeline to the AllWeather setting
(Outdoor-Rain + RainDrop + Snow100K), trained on a single mixed folder of
paired images and evaluated on the three standard test sets with
PSNR / SSIM / MAE / LPIPS / DISTS.

## Files

- `train_allweather.py` — training (Lightning, same recipe as `train.py`: L1 + 0.1·FFT loss, AdamW, linear warmup + cosine annealing, random crop + flip/rotation augmentation).
- `test_allweather.py` — inference on the three test sets + metric computation.
- `utils/allweather_utils.py` — datasets for the folder layouts below.
- `utils/full_metrics.py` — metric code ported from the reference `metrics/` package (Y-channel PSNR/SSIM/MAE per the SwinIR convention, LPIPS-AlexNet, DISTS from `piq`), so numbers are directly comparable.

## Extra dependencies

On top of the environment in `env.yaml`:

```
pip install lpips piq
```

## Expected data layout

Training set (mixed):

```
allweather/
├── input/     degraded images
├── gt/        clean images (same filenames as input/)
└── gt_val/    (optional) secondary gt folder
```

Consistent with the StarIR paper, **no validation split is used**: the full
training set is trained on. `gt_val/` is treated as a secondary gt source —
an input is first matched against `gt/`, then against `gt_val/` — so every
image is included whether the files in `gt_val/` are copies of files still
in `gt/` or were moved out of it. Checkpoints are evaluated afterwards with
`test_allweather.py`.

Test sets:

```
CVPR19RainTrain/test/                 → data/ + gt/        (Outdoor-Rain)
raindrop_data/test_a/                 → data/ + gt/        (RainDrop test_a)
Snow100K-testset/.../test/Snow100K-L/ → synthetic/ + gt/   (Snow100K-L)
```

Degraded→gt filename matching handles the three naming schemes automatically:
identical names (Snow100K), `X_rain.png → X_clean.png` (RainDrop), and
`im_0001_s80_a04.png → im_0001.png` (Outdoor-Rain). Unmatched files are
reported at startup — if you see unmatched examples printed, check the naming.

## Training

```
cd All_in_One
python train_allweather.py --data_dir ~/autodl-tmp/allweather --num_gpus 1 --batch_size 16
```

Useful flags:

- `--epochs 150 --warmup_epochs 15 --lr 2e-4 --patch_size 128` (defaults, following the all-in-one recipe)
- `--ckpt_every 10` — periodic checkpoints in `--ckpt_dir` (default `ckpt/allweather`); `last.ckpt` is always kept
- `--resume ckpt/allweather/last.ckpt` — resume training
- `--num_gpus 2` — DDP across GPUs (batch size is per GPU)

TensorBoard logs are written to `logs/StarIR-AllWeather/`.

## Testing / evaluation

```
cd All_in_One
python test_allweather.py --ckpt ckpt/allweather/last.ckpt \
    --outdoor_rain_dir ~/autodl-tmp/test/CVPR19RainTrain/test \
    --raindrop_dir     ~/autodl-tmp/test/raindrop_data/test_a \
    --snow100k_dir     ~/autodl-tmp/test/Snow100K-testset/jdway/GameSSD/overlapping/test/Snow100K-L \
    --output_path results_allweather/
```

Any of the three `--*_dir` options can be omitted to test a subset of tasks.

For each task this saves restored images to `results_allweather/<task>/`,
writes `metrics.txt` (mean ± std of the 5 metrics, plus per-image inference
time), and finally writes a combined `results_allweather/metrics_summary.txt`
with all results as `mean ± std`, the inference time per task, and the model
parameter count. Predictions are saved to disk as PNG and read back for
evaluation, matching the reference measuring pipeline exactly (metrics are
computed on 8-bit images).

Inference time is measured per image around the padded forward pass only
(CUDA-synchronized; data loading and PNG writing excluded), and the first
image of each task is excluded from the statistics since it absorbs warm-up
overhead. With `--skip_inference` no timing is available and the time column
reads `n/a`.

`--skip_inference` re-evaluates previously saved outputs without re-running
the model; `--verbose` prints per-image metrics.

## Notes

- Inference pads inputs to a multiple of 32 (reflect padding) and crops back,
  as in `test.py`; no cropping of gt is performed, so saved outputs align
  exactly with the original gt images.
- The checkpoint loader accepts Lightning checkpoints from
  `train_allweather.py` / `train.py` (strips the `net.` prefix) as well as
  plain state dicts and basicsr-style `params`/`params_ema` checkpoints.
