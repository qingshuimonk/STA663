# STA663 Class Project - Variational Auto-encoder
![VAE Demo](https://github.com/qingshuimonk/STA663/blob/master/img/vae_demo.png "VAE Demo")
  
This repo contains final project of Duke STA663. A [variational auto-encoder](https://arxiv.org/pdf/1606.05908.pdf) is implemented in this project. Our work includes:
1. An implementation of Variational Auto-encoder using [Tensorflow](https://www.tensorflow.org/)
2. Using CPU parallel to optimize the code
3. Discussion about using [Numba](http://numba.pydata.org/) and [Cython](http://cython.org/) for bottleneck in vae
4. Comparison of two other frameworks: Auto-encoder and Generative Adversarial Nets  

For demonstration purpose, we use [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) to test our autoencoder.

# Installation
#### Install Conda 
It's hard to install Tensorflow in different OS, we highly recommend using Anaconda
To install Anaconda, see instructions [here](https://conda.io/docs/install/quick.html)
#### Pull Git Repo
```bash
git clone https://github.com/qingshuimonk/STA663.git
```
#### Install Conda Environment
```
cd STA663
conda env create
source activate vae-env (activate vae-env if using Windows)
```
#### Install Tensorflow Via Conda
```
conda install -c conda-forge tensorflow=1.0.0
```
#### Install Package
```
python setup.py install
```
#### Run Demo
##### 1. VAE
Run an Variational Autoencoder that generate synthetic data from [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
```
python demo_vae.py
```
If you ran into a 'PyQt4' issue with matplotlib when running this, try uncomment line 7 in demo_vae.py:
```python
matplotlib.use('PS')
```
##### 2. Normal AE
This runs a normal auto-encoder with two hidden layers (256 * 128)
```
python ae.py
```
This runs a normal auto-encoder with same structure of our VAE's default configuration (500 * 500)
```
python ae_same_structure.py
```
##### 3. GAN
Warning: GAN takes a really long time (almost half an hour on my Mac)
```
python Vanilla_GAN.py
```

# Other Demos
- [Implementation of VAE & Unit Tests](https://github.com/qingshuimonk/STA663/blob/master/docs/vae_unit.ipynb)
- Using CPU parallel to optimize the code [1](https://github.com/qingshuimonk/STA663/blob/master/docs/runtime_cmp.ipynb), [2](https://github.com/qingshuimonk/STA663/blob/master/docs/runtime_cmp_parallel.ipynb)
- [Optimize bottleneck using Numba and Cython](https://github.com/qingshuimonk/STA663/blob/master/docs/optimize_forward_scale2.ipynb)
- [A demo of using Auto-encoder](https://github.com/qingshuimonk/STA663/blob/master/docs/Autoencoder.ipynb)
- [A demo of using GAN](https://github.com/qingshuimonk/STA663/blob/master/docs/Vanilla_GAN.ipynb)


# TO-DO
- [x] Make code files for Variational Autoencoder
- [x] Optimize reconstruct function 
- [x] Compare running time for raw vae and optimized vae
- [x] Compare with other algorithms
- [x] Wrap up codes
- [ ] Write Report