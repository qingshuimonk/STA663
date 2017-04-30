# STA663
Final project of STA 663

# Install
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
Run an Variational Autoencoder that generate synthetic data from [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
```
python demo_vae.py
```


# TO-DO
- [x] Make code files for Variational Autoencoder
- [x] Optimize reconstruct function 
- [x] Compare running time for raw vae and optimized vae
- [x] Compare with other algorithms
- [x] Wrap up codes

# Structure
- Put all tempororay files in ]scripts, this folder will not be syncronized
- Put final files in root directory