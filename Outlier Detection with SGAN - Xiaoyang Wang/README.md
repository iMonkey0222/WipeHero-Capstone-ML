##  Semi-Supervised Learning with GAN in Keras

This outlier detection project designed for WipeHero is implemented by semi-supervised learning GAN, here is called SGAN, which is a re-implementation of this paper [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583). In addition, the idea of SGAN model structure borrowed much from [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN). The improvement and modification is based on it.
The motivation of this project is to recognize outliers from images provided by car washers of company WipeHero. In oder to achive this result and considering the data situation that there is only limited quality human labelled data, semi-supervised learning GAN was chosen to solve it via classification task.

## Implementation Description
The original data for outliers detection is unlabelled, unbalanced and there is not enough outliers' data.
- First preprocessing the data and augmenting data 
  - Here I only consider category: **outside front**. Since only part of this category's data was labelled
  - There are 2 classes: **class-0: noraml cars. class-1: outleirs**.
- Train and test SGAN model with 1000 labelled data (800 for train, 200 for test)
- Generative resultes and Prediction Test results 

## Prerequisites
- Jupyter Notebook
- Python 3
- Tensorflow CPU version, Theano
- Numpy
- Scipy
- Skimage
- PIL
- H5PY
- Keras

## Processed & Augmented Dataset Sample Review
![dataset](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/1.Input%20Processing%20and%20Outliers%20Augmenetation/processed_dataset_samples.png?raw=true)

## Resultes
### Classifier Accuracy
The classifier accuracy of the model can achieve 80.98% if not considering the fake class.
![accuracy](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/3.Results/Details%20of%20loss/experiments.png?raw=true)

### Generative Results
- Generated Samples 1000th epochs
![1000epoch](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/3.Results/Generative%20Results/outsideFront_epoch1000.png?raw=true)

- Generated Samples 2000th iterations
![2000epochs](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/3.Results/Generative%20Results/outsideFront_epoch2000.png?raw=true)

## Training Details

![loss](https://github.com/iMonkey0222/WipeHero-Capstone-ML/blob/master/Outlier%20Detection%20with%20SGAN%20-%20Xiaoyang%20Wang/3.Results/Images%20of%20loss_lr_experiments/loss.png?raw=true)
