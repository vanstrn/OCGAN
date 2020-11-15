# OneClassGAN

Creating the environment

`conda env create -f environment.yml`

Will create a conda environment `ocgan`, which should be compatible with GPU.
If this doesn't work try installing mxnet manually with:

`pip install mxnet-cu101==1.4.1`

Running the training:

`python2 TrainNovelty.py --expname experiment1 --dataset MNIST --ngf 64 --ndf 12 --lambda 10 --datapath ~/OCGAN/mnist_png/mnist_png/ --noisevar 0.2 --classes 1 --latent 16 --gpu_num 0`

- `datapath` - points to a folder with data folder.
- `dataset` - points to a specific folder of data, which contains nested folders representing each class.
- `classes` - which class to leave out (I modified this to leave it out of the training set.)
- `gpu_num` - which gpu to run the operations on.

Evaluation:

`python2 TestNovelty.py --expname experiment1 --dataset MNIST --ngf 64 --ndf 12 --datapath ~/OCGAN/mnist_png/mnist_png/ --noisevar 0.2 --latent 16`


Before running trials you will need to create output and checkpoint folders:

`mkdir outputs`

`mkdir checkpoints`
