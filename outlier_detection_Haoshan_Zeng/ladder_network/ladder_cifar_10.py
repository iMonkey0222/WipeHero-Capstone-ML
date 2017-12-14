#!/usr/bin/python3

from __future__ import print_function

import numpy as np
import argparse
import os

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from encoder import StackedEncoders
from decoder import StackedDecoders

class Ladder(torch.nn.Module):
    def __init__(self, encoder_sizes, decoder_sizes, 
                 encoder_activations, encoder_train_bn_scaling, noise_std, use_cuda):
        super(Ladder, self).__init__()
        self.use_cuda = use_cuda
        decoder_in = encoder_sizes[-1]
        encoder_in = decoder_sizes[-1]
        self.StackE = StackedEncoders(encoder_in, encoder_sizes, encoder_activations,
                                     encoder_train_bn_scaling, noise_std, use_cuda)
        self.StackD = StackedDecoders(decoder_in, decoder_sizes, encoder_in, use_cuda)
        self.batch_image = torch.nn.BatchNorm2d(encoder_in, affine=False)
        
    def forward_encoders_clean(self, data):
        return self.StackE.forward_clean(data)
    
    def forward_encoders_noise(self, data):
        return self.StackE.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.StackD.forward(tilde_z_layers, encoder_output, tilde_z_bottom)
    
    def get_encoders_tilde_z(self, reverse=True):
        return self.StackE.get_encoders_tilde_z(reverse)
    
    def get_encoders_z_pre(self, reverse=True):
        return self.StackE.get_encoders_z_pre(reverse)
    
    def get_encoder_tilde_z_bottom(self):
        return self.StackE.buffer_tilde_z_bottom.clone()
    
    def get_encoders_z(self, reverse=True):
        return self.StackE.get_encoders_z(reverse)
    
    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.StackD.bn_hat_z_layers(hat_z_layers, z_pre_layers)
    
    

def evaluate_performance(ladder, valid_loader, epoch, agg_cost_scaled, 
                         agg_supervised_cost_scaled, agg_unsupervised_cost_scaled, cuda):
    correct = 0.
    total = 0.
    # enumerate valid_loader
    for batch_idx, (data, target) in enumerate(valid_loader):
        if cuda:
            data = data.cuda()
        data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)
        
        if cuda:
            output = output.cpu()
            target = target.cpu()
        output = output.data.numpy()
        prediction = np.argmax(output, axis=1)
        target = target.data.numpy()
        correct += np.sum(target == prediction)
        total += target.shape[0]
    
    print('epoch: ', epoch+1, '\t',
         'total cost: ', '{:.4f}'.format(agg_cost_scaled), '\t',
         'supervised cost: ', '{:.4f}'.format(agg_supervised_cost_scaled), '\t',
         'unsupervised cost: ', '{:.4f}'.format(agg_unsupervised_cost_scaled), '\t',
         'validation accuracy: ', correct/total)
    
import keras
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)

x_train = x_train.astype(float)
x_test = x_test.astype(float)
x_train = np.multiply(x_train, 1./255.)
x_test = np.multiply(x_test, 1./255.)

# y_train = keras.utils.to_categorical(y_train, 10).astype(int)
# y_test = keras.utils.to_categorical(y_test, 10).astype(int)

y_train = y_train.astype(int).reshape(y_train.shape[0])
y_test = y_test.astype(int).reshape(y_test.shape[0])

X_train_label = x_train[:5000]
Y_train_label = y_train[:5000]
X_train_unlabel = x_train[5000:]
Y_train_unlabel = y_train[5000:]
X_test = x_test
Y_test = y_test

X_train_unlabel = X_train_unlabel.reshape(X_train_unlabel.shape[0],32*32*3)
X_train_label = X_train_label.reshape(X_train_label.shape[0],32*32*3)
X_test = X_test.reshape(X_test.shape[0],32*32*3)

# X_train_label = np.multiply(X_train_label, 1./255.)
# X_test = np.multiply(X_test, 1./255.)

# X_train_unlabel = X_train_label[300:]
# Y_train_unlabel = Y_train_label[300:]
# X_train_label = X_train_label[:300]
# Y_train_label = Y_train_label[:300]

# X_train_unlabel = X_train_unlabel.reshape(X_train_unlabel.shape[0],64*64*3)
# X_train_label = X_train_label.reshape(X_train_label.shape[0],64*64*3)

# X_test = X_test.reshape(X_test.shape[0],64*64*3)




print(X_train_unlabel.shape, Y_train_unlabel.shape)
print(X_train_label.shape, X_test.shape)
print(Y_train_label.shape, Y_test.shape)


def run_ladder(X_train_label, Y_train_label, batch_size=100, epochs=10, noise_std=0.2, data_dir='data', seed=42,
              u_costs='0.1, 0.1, 0.1, 0.1, 0.1, 10., 1000.', cuda=False, decay_epoch=15):
    if cuda and not torch.cuda.is_available():
        print('Torch.cuda is not available, using CPU. \n')
        cuda=False
    
    print('=========================================')
    print('batch size: ', batch_size)
    print('epochs: ', epochs)
    print('random seed: ', seed)
    print('noise std: ', noise_std)
    print('decay epoch: ', decay_epoch)
    print('cuda: ', cuda)
    print('=========================================\n')
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    
    print('Data Loading...')
    dataset_unlabel = TensorDataset(torch.FloatTensor(X_train_unlabel), torch.LongTensor(Y_train_unlabel))
    loader_unlabel = DataLoader(dataset_unlabel, batch_size=batch_size, shuffle=True, **kwargs)
    dataset_test = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(Y_test))
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, **kwargs)
    
    # Configure the Ladder
    starter_lr = 0.02
    encoder_sizes = [1000, 500, 250, 250, 250, 10]
    decoder_sizes = [250, 250, 250, 500, 1000, 32*32*3]
    unsupervised_costs_lambda = [float(x) for x in u_costs.split(',')]
    encoder_activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']
    encoder_train_bn_scaling = [False, False, False, False, False, True]
    ladder = Ladder(encoder_sizes, decoder_sizes, encoder_activations,
                    encoder_train_bn_scaling, noise_std, cuda)
    optimizer = Adam(ladder.parameters(), lr=starter_lr)
    loss_supervised = torch.nn.CrossEntropyLoss()
    loss_unsupervised = torch.nn.MSELoss()
    
    if cuda:
        ladder.cuda()
        
    assert len(unsupervised_costs_lambda) == len(decoder_sizes) + 1
    assert len(encoder_sizes) == len(decoder_sizes)
    
    print('')
    print('========network=======')
    print(ladder)
    print('======================')

    print('')
    print('== unsupervised cost ==')
    print(unsupervised_costs_lambda)

    print('')
    print('=====================')
    print('training\n')
    
    

    # Add annealing of learning rate after 100 epochs
    for e in range(epochs):
        agg_cost = 0.
        agg_supervised_cost = 0.
        agg_unsupervised_cost = 0.
        num_batches = 0
        ladder.train()
        # print(X_train_label.shape[0])
        # Add volatile for the input parameters in training and validation
        ind_labelled = 0
        ind_limit = np.ceil(float(X_train_label.shape[0]) / batch_size)
        
        if e > decay_epoch:
            ratio = float(epochs - e) / (epochs - decay_epoch)
            current_lr = starter_lr * ratio
            optimizer = Adam(ladder.parameters(), lr=current_lr)
            
        for batch_idx, (unlabelled_images, unlabelled_labels) in enumerate(loader_unlabel):
            if ind_labelled == ind_limit:
                randomize = np.arange(X_train_label.shape[0])
                np.random.shuffle(randomize)
                X_train_label = X_train_label[randomize]
                Y_train_label = Y_train_label[randomize]
                ind_labelled = 0
                
            # Verify whether labelled examples are used for calculating unsupervised loss
            labelled_start = batch_size * ind_labelled
            labelled_end = batch_size * (ind_labelled + 1)
            ind_labelled += 1
            batch_X_train_label = torch.FloatTensor(X_train_label[labelled_start:labelled_end])
            batch_Y_train_label = torch.LongTensor(Y_train_label[labelled_start:labelled_end])
            # print(batch_X_train_label.shape, batch_Y_train_label.shape)
            
            if cuda:
                batch_X_train_label = batch_X_train_label.cuda()
                batch_Y_train_label = batch_Y_train_label.cuda()
                unlabelled_images = unlabelled_images.cuda()
                
            labelled_data = Variable(batch_X_train_label, requires_grad=False)
            labelled_target = Variable(batch_Y_train_label, requires_grad=False)
            unlabelled_data = Variable(unlabelled_images)
            
            optimizer.zero_grad()
            
            # do a noisy pass for labelled data
            output_noise_labelled = ladder.forward_encoders_noise(labelled_data)

            # do a noisy pass for unlabelled_data
            output_noise_unlabelled = ladder.forward_encoders_noise(unlabelled_data)
            tilde_z_layers_unlabelled = ladder.get_encoders_tilde_z(reverse=True)
            
            # do a clean pass for unlabelled data
            output_clean_unlabelled = ladder.forward_encoders_clean(unlabelled_data)
            z_pre_layers_unlabelled = ladder.get_encoders_z_pre(reverse=True)
            z_layers_unlabelled = ladder.get_encoders_z(reverse=True)

            tilde_z_bottom_unlabelled = ladder.get_encoder_tilde_z_bottom()
            
            # pass through decoders
            hat_z_layers_unlabelled = ladder.forward_decoders(tilde_z_layers_unlabelled,
                                                              output_noise_unlabelled,
                                                              tilde_z_bottom_unlabelled)
            
            z_pre_layers_unlabelled.append(unlabelled_data)
            z_layers_unlabelled.append(unlabelled_data)
            
            # batch normalize using mean, var of z_pre
            bn_hat_z_layers_unlabelled = ladder.decoder_bn_hat_z_layers(hat_z_layers_unlabelled, 
                                                                        z_pre_layers_unlabelled)
            
            # calculate costs
            cost_supervised = loss_supervised.forward(output_noise_labelled, labelled_target)
            cost_unsupervised = 0.
            assert len(z_layers_unlabelled) == len(bn_hat_z_layers_unlabelled)
            for cost_lambda, z, bn_hat_z in zip(unsupervised_costs_lambda, z_layers_unlabelled, 
                                                bn_hat_z_layers_unlabelled):
                c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
                cost_unsupervised += c
                
            # backprop
            cost = cost_supervised + cost_unsupervised
            cost.backward()
            optimizer.step()
            
            agg_cost += cost.data[0]
            agg_supervised_cost += cost_supervised.data[0]
            agg_unsupervised_cost += cost_unsupervised.data[0]
            num_batches += 1

            if ind_labelled == ind_limit:
                # Evaluation
                ladder.eval()
                evaluate_performance(ladder, loader_test, e,
                                     agg_cost / num_batches,
                                     agg_supervised_cost / num_batches,
                                     agg_unsupervised_cost / num_batches,
                                     cuda=False)
                ladder.train()
    print("=====================\n")
    print("Done :)")

run_ladder(X_train_label, Y_train_label, batch_size=100, epochs=50, noise_std=0.2, data_dir='data', seed=42,
              u_costs='0.1, 0.1, 0.1, 0.1, 0.1, 10., 1000.', cuda=False, decay_epoch=20)