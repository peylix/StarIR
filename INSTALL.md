## Installation

conda create -n starir python=3.9

conda activate starir

conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch


pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort

python setup.py develop --no_cuda_ext


