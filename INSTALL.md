## Installation

conda create -n starir python=3.9

conda activate starir

conda install pytorch=2.4.0 torchvision pytorch-cuda=12.4 -c pytorch


pip install opencv-python lmdb tqdm einops scipy scikit-image tensorboard natsort

python setup.py develop --no_cuda_ext



Dataset [download](https://drive.google.com/file/d/1JYk08xMeI28wN_EUehvI7R4dqGA-hb3q/view?usp=sharing)

For trianing and testing, your directory structure should look like this

`Datasets` <br/>
     `└──LOL-v2` <br/>
          `└──Synthetic` <br/>
               `├──Train`  <br/>
                    `├──Normal`  <br/>
                    `└──Low`  
               `└──Test`  <br/>
                    `├──Normal`  <br/>
                    `└──Low`  


