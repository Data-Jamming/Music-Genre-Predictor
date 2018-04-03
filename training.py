from rnn import rnn
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import dataset
import nltk
import OHencoder
import random
import os

PRINTERVAL = 1
BATCH_SIZE = 1 #For now, keep it as 1 each song is a batch for each word in the song
MAX_EPOCHS = 1000
LEARNING_RATE = 0.001
VAL_RATIO = 0.1

LOG_PATH = "logs/checkpoint02.pth"
SAVE_PATH = "save/best_model02.pth"

class Trainer():
    def __init__(self):
        csv_reader = dataset.load_data("dataset/cleaned_lyrics.csv")

        next(csv_reader)

        for i in range(0): #TEMPORARY
            next(csv_reader)

        self.data = []
        self.val_data = []
        self.labels = []
        self.val_labels = []

        datasize = 1000 #Just make this arbitrarily large when you want to use the whole dataset

        print("Loading data...")
        for i in range(datasize):
            try:
                song = next(csv_reader)
                self.data.append(nltk.word_tokenize(song[2]))
                self.labels.append(song[1])
            except StopIteration:
                break;

        self.val_data = self.data[int(len(self.data)*(1-VAL_RATIO)):]
        self.val_labels = self.labels[int(len(self.data)*(1-VAL_RATIO)):]
                
        print("Building encoder...")
        self.data_encoder = OHencoder.encode(j for i in self.data for j in i)
        self.label_encoder = OHencoder.encode(self.labels)

        self.data_decoder = list(self.data_encoder.keys())  #Gives you word/genre from vector index
        self.label_decoder = list(self.label_encoder.keys())

        self.data = self.data[:int(len(self.data)*(1-VAL_RATIO))]
        self.labels = self.labels[:int(len(self.labels)*(1-VAL_RATIO))]
        
        #print([data_enconder[word] for word in data[-1]])
        self.model = rnn(len(self.data_encoder), [32], [32], len(self.label_encoder))
        self.best_acc = 0
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)

    def get_pred(self, lyrics):
        rec = self.model.init_rec()
        preds = Variable(torch.FloatTensor(len(lyrics), self.model.n_classes).zero_())
        for i, word in enumerate(lyrics):
            #print(word)
            input = Variable(torch.zeros(1, len(self.data_encoder)))
            
            input[0, self.data_encoder[word]] = 1
            pred, rec = self.model.forward(input, rec)
            preds[i] = pred

        return preds

    
    def get_accuracy(self):
        correct = 0
        for i, song in enumerate(self.val_data):
            pred = self.get_pred(song)
            value, index = torch.max(torch.mean(pred, 0, True), 1)
            if self.label_encoder[self.val_labels[i]] == int(index):
                correct += 1
                
        acc = correct/len(self.val_data)
        
        if acc > self.best_acc:
            self.best_acc = acc
            
        return acc
    
    def save_checkpoint(self, epoch, isbest=False, path=LOG_PATH):
        torch.save({
            'epoch':epoch,
            'state_dict':self.model.state_dict(),
            'best_acc':self.best_acc,
            'optimizer':self.optimizer.state_dict()
        }, path)
        if isbest:
            torch.save({
                'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'best_acc':self.best_acc,
                'optimizer':self.optimizer.state_dict()
        }, SAVE_PATH)
        
    def train(self):

        #for param in self.model.parameters():
        #    print(param.size())
        
        
        songs = [[self.data[i], self.labels[i]] for i in range(len(self.data))]
        random.shuffle(songs)
        preds = Variable(torch.FloatTensor(BATCH_SIZE, len(self.label_encoder)).zero_())
        labels = Variable(torch.LongTensor(BATCH_SIZE).zero_())
        for i, song in enumerate(songs):
            lyrics = song[0]
            genre = song[1]
            pred = self.get_pred(lyrics)
            labels = Variable(torch.LongTensor([self.label_encoder[genre] for i in pred]))

            preds = pred
            
            if (i+1)%PRINTERVAL == 0:
                value, index = torch.max(torch.mean(pred, 0, True), 1)
                #print(preds[i])
                #print(labels[i])
                print(str(i)+ ":", "Guessing", self.label_decoder[int(index[0])], "With", float(value[0]), "confidence. Correct genre was", genre + ".")

            if (i+1)%BATCH_SIZE == 0 or i >= len(songs)-1:
                self.loss = self.criterion(preds, labels)
                
                self.optimizer.zero_grad()

                self.loss.backward()

                self.optimizer.step()
                preds = Variable(torch.FloatTensor(BATCH_SIZE, len(self.label_encoder)).zero_())
                labels = Variable(torch.LongTensor(BATCH_SIZE).zero_())
                
trainer = Trainer()    
epoch_start = 0
if os.path.isfile(LOG_PATH):
    print("Checkpoint found! Resuming...")
    checkpoint = torch.load(LOG_PATH)
    epoch_start = checkpoint['epoch']
    trainer.model.load_state_dict(checkpoint['state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.best_acc = checkpoint['best_acc']
    

for i in range(epoch_start, MAX_EPOCHS):
    print("\nEpoch", i+1)
    trainer.train()
    acc = trainer.get_accuracy()
    print("\nLoss:",trainer.loss.data[0],"\nAccuracy:", acc)
    trainer.save_checkpoint(i+1, isbest=(acc == trainer.best_acc))


