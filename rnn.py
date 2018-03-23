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
    def __init__(self, n_input, out_layers_sizes, rec_layers_sizes, n_classes):
        """
        Params:
            n_input (int): size of the input vector.
            out_layers_sizes (list): sizes of each hidden layer.
            rec_layers_sizes (list): sizes of each recurrence layer.
            n_classes (int): number of classes (music genres, in our case).
        """
        super(rnn, self).__init__()

        # we want a hidden state for each layer
        assert len(out_layers_sizes) == len(rec_layers_sizes)
        self.num_layers = len(out_layers_sizes)

        self.n_classes = n_classes

        # I'm not really sure how big this should be I just picked arbitrarily.
        # We might want it to be at least the size of n_classes to say how likely it thinks each class is
        # but maybe it should try to compress that info?
        self.hidden_state_size = n_classes

        self.n_input = n_input

        self.out_layers = []
        self.out_layers.append(nn.Linear(n_input + self.hidden_state_size, out_layers_sizes[0]))
        for i, n_layer in enumerate(out_layers_sizes[:-1]):
            self.out_layers.append(nn.Linear(n_layer+self.hidden_state_size, out_layers_sizes[i+1]))
        self.out_layers.append(nn.Linear(out_layers_sizes[-1]+self.hidden_state_size, self.n_classes))

        # I'm thinking rec_layers_sizes should be really small probably only one layer???
        self.rec_layers=[]
        self.rec_layers.append(nn.Linear(n_input + self.hidden_state_size, rec_layers_sizes[0]))
        for i, n_layer in enumerate(rec_layers_sizes[:-1]):
            self.rec_layers.append(nn.Linear(n_layer+self.hidden_state_size, rec_layers_sizes[i+1]))
        self.rec_layers.append(nn.Linear(rec_layers_sizes[-1]+self.hidden_state_size, self.hidden_state_size))

        # we have to register our NN params manually bc we init'd them within the append method
        # this tells our optimizer (created in training) what parameters it must optimize
        for i, layer in enumerate(self.out_layers + self.rec_layers):
            self.register_parameter("l" + str(i), layer.weight)
            self.register_parameter("b" + str(i), layer.bias)

        self.init_hidden_states()

    def forward(self, input_word):
        """
        Params:
            input_word (tensor): one-hot encoding of a word.
        Returns:
        """
        layer_input = input_word

        # feed input into both neural networks, layer by layer
        # update hidden states
        for i in range(self.num_layers):
            cur_layer, cur_hidden_layer = self.out_layers[i], self.rec_layers[i]

            layer_input = torch.cat(
                (layer_input, self.hidden_states[i]), dim=1
            )
            layer_input_hs = torch.cat(
                (layer_input_hs, self.hidden_states[i]), dim=1
            )

            # relu is applied element-wise, so the shape is conserved
            layer_input = F.relu(cur_layer(layer_input))
            layer_input_hs = F.relu(cur_hidden_layer(layer_input_hs))

            self.hidden_states[i] = layer_input_hs

        #print("output:", output)
        #print("rec:", rec)

        # return the output of the last layer
        return layer_input

    def init_rec(self):
        return Variable(torch.zeros(1, self.hidden_state_size))

    def init_hidden_states(self):
        # we create a hidden state for each layer, init at zero
        self.hidden_states = Variable(torch.zeros(self.num_layers, self.hidden_state_size))