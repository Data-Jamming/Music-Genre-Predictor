from rnn import rnn
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import dataset
import nltk
import OHencoder
import random

PRINTERVAL = 1

class Trainer():
    def __init__(self):
        csv_reader = dataset.load_data("dataset/cleaned_lyrics.csv")

        next(csv_reader)

        self.data = []
        self.labels = []

        datasize = 100 #Just make this arbitrarily large when you want to use the whole dataset
        
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

        #print([data_enconder[word] for word in data[-1]])
    
        self.model = rnn(len(self.data_encoder), [64, 64], [64, 64], len(self.label_encoder))
    
    def get_pred(self, lyrics):
        rec = self.model.init_rec()
        for word in lyrics:
            input = Variable(torch.zeros(1, len(self.data_encoder)))
            input[0, self.data_encoder[word]] = 1
            pred, rec = self.model.forward(input, rec)
        return pred
        
    def train(self):
        criterion = nn.NLLLoss()
        o = torch.optim.SGD(self.model.parameters(), lr = 0.001)
        songs = [[x, y] for x in self.data for y in self.labels]
        random.shuffle(songs)
        for i, song in enumerate(songs):
            lyrics = song[0]
            genre = song[1]
            pred = self.get_pred(lyrics)
            y = Variable(torch.LongTensor([self.label_encoder[genre]]))
            
            if i%PRINTERVAL == 0:
                value, index = torch.max(pred, 1)
                print(str(i)+ ":", "Guessing", self.label_decoder[int(index[0])], "With", int(value[0]), "confidence. Correct genre was", genre + ".")
            
            loss = criterion(pred, y)
            loss.backward()

trainer = Trainer()
trainer.train()



