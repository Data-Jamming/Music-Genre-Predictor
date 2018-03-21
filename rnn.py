#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

# I was semi following this tutorial:
# http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


class rnn(nn.Module):
    # Describe the shape of the neural net, n_input will be the size of the vector
    # layers are arrays where each element is an integer that describes the size of that hidden layer
    # the out is what the NN will say what class it is and the rec is the part that gets "recurred"
    # to the next iteration of the rnn.
    # n_classes is the number of things we are classifying.
    def __init__(self, n_input, out_layers, rec_layers, n_classes):
        super(rnn, self).__init__()
        self.n_classes = n_classes
        self.rec_size = n_classes  # I'm not really sure how big this should be I just picked arbitrarily.
        #We might want it to be at least the size of n_classes to say how likely it thinks each class is 
        #but maybe it should try to compress that info?
        self.n_input = n_input
        self.softmax = nn.LogSoftmax(dim=0)
        self.out_layers = []
        self.out_layers.append(nn.Linear(n_input + self.rec_size, out_layers[0]))
        for i, n_layer in enumerate(out_layers[:-1]):
            self.out_layers.append(nn.Linear(n_layer, out_layers[i+1]))
        self.out_layers.append(nn.Linear(out_layers[-1], self.n_classes))
        # I'm thinking rec_layers should be really small probably only one layer???
        self.rec_layers=[]
        self.rec_layers.append(nn.Linear(n_input + self.rec_size, rec_layers[0]))
        for i, n_layer in enumerate(rec_layers[:-1]):
                self.rec_layers.append(nn.Linear(n_layer, rec_layers[i+1]))

        self.rec_layers.append(nn.Linear(rec_layers[-1], self.rec_size))
        
        for i, layer in enumerate(self.out_layers + self.rec_layers):
            self.register_parameter("l" + str(i), layer.weight)
            self.register_parameter("b" + str(i), layer.bias)

    def forward(self, input, rec):
        combine=torch.cat((input, rec), 1)
        
        output=F.relu(self.out_layers[0](combine))
        for l in self.out_layers[1:-1]:
            output=F.tanh(l(output))
        output=F.relu(self.out_layers[-1](output))
        #output=self.softmax(output)

        rec=F.relu(self.rec_layers[0](combine))
        for l in self.rec_layers[1:]:
            rec=F.relu(l(rec))

        print("output:", output)
        print("rec:", rec)
            
        return output, rec

    def init_rec(self):
        return Variable(torch.zeros(1, self.rec_size))
