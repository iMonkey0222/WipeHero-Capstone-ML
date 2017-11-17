##  Semi-Supervised Learning with GAN in Keras

This outlier detection project designed for WipeHero is implemented by semi-supervised learning GAN, here is called SGAN, which is a re-implementation of this paper [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583). In addition, the idea of SGAN model structure borrow much from [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN). The improvement and modification is based 
on it.

### Implementation Description
- First preprocessing the data and augmenting data
- Train and test SGAN model with 1000 labelled data (800 for train, 200 for test)
- Generative resultes and Prediction Test results 


### Prerequisites
- Jupyter Notebook
- Python 3
- Tensorflow CPU version, Theano
- Numpy
- Scipy
- Skimage
- PIL
- H5PY
- Keras

### Classifier Accuracy
The model can achieve 80.98% classifier accuracy if not considering the fake class.


