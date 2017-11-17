##  Semi-Supervised Learning with GAN in Keras

This outlier detection project designed for WipeHero is implemented by semi-supervised learning GAN, here is called SGAN, which is a re-implementation of this paper [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583). In addition, the idea of SGAN model structure borrowed much from [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN). The improvement and modification is based 
on it.

### Implementation Description
- First preprocessing the data and augmenting data (here only considering category: **outside front**)
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

### Processed & Augmented Dataset Sample Review
![dataset](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/1.Input%20Processing%20and%20Outliers%20Augmenetation/processed_dataset_samples.png?raw=true)

### Classifier Accuracy
The classifier accuracy of the model can achieve 80.98% if not considering the fake class.

### Generative Results
- Model train 1000 iterations
![1000epoch](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/3.Results/Generative%20Results/outsideFront_epoch1000.png?raw=true)

- Model train 2000 iterations
![2000epochs](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/3.Results/Generative%20Results/outsideFront_epoch2000.png?raw=true)

