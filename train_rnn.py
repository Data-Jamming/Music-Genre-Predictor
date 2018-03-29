from rnn import rnn
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import dataset
import nltk
import OHencoder
import random

from make_rnn_great_for_once import MyRNN

PRINTERVAL = 1
BATCH_SIZE = 1

class Trainer():
    def __init__(self):
        csv_reader = dataset.load_data("dataset/cleaned_lyrics.csv")

        # skip past column headers
        next(csv_reader)

        for i in range(0): #TEMPORARY
            next(csv_reader)

        self.data = []
        self.labels = []

        datasize = 1000 #Just make this arbitrarily large when you want to use the whole dataset

        print("Loading data...")
        for i in range(datasize):
            try:
                song = next(csv_reader)
                self.data.append(nltk.word_tokenize(song[2]))
                self.labels.append(song[1])
            except StopIteration:
                break;

        print("Building encoder...")
        self.data_encoder = OHencoder.encode(j for i in self.data for j in i)
        self.label_encoder = OHencoder.encode(self.labels)

        self.data_decoder = list(self.data_encoder.keys())  #Gives you word/genre from vector index
        self.label_decoder = list(self.label_encoder.keys())

        self.model = MyRNN(len(self.data_encoder))


    # NOTE: not tested yet!
    # does not pass initial hidden state, which defaults to zero
    def get_pred(self, lyrics):
        one_hot_lyrics = []
        for word in lyrics:
            # one-hot encode
            one_hot_word = Variable(torch.zeros(1, len(self.data_encoder)))
            one_hot_word[0, self.data_encoder[word]] = 1
            one_hot_lyrics[] = one_hot_word
        # Must be of shape (LYRIC_TOTAL, BATCH_SIZE, ONE_HOT_WORD_SIZE)
        model_input = torch.FloatTensor([one_hot_lyrics])
        pred, _ = self.model(model_input)

    # TODO: implement
    def train(self):
        pass

trainer = Trainer()
trainer.train()