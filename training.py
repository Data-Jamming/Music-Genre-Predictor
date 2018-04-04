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
LEARNING_RATE = 0.002
VAL_RATIO = 0.1
SAVE_INTERVAL = 1000

STRATIFY_DATA = True
# arbitrary!
NUM_GENRES = 10

LOG_PATH = "logs/checkpoint.pth"
SAVE_PATH = "save/best_model.pth"

class Trainer():
    def __init__(self):
        datasize = 250 #Just make this arbitrarily large when you want to use the whole dataset
        print("Loading data...")
        if STRATIFY_DATA:
            self.data, self.labels = dataset.get_stratified_data(datasize)
        else:
            self.data, self.labels = dataset.get_data(datasize)

        print("Building encoder...")

        self.data_encoder = OHencoder.map_to_int_ids(self.data, threshold = 4)
        self.label_encoder = OHencoder.map_to_int_ids([self.labels])

        for x in self.data_encoder:
            self.data_encoder[x] += 1 
            
        shuffle = [i for i in range(len(self.data))]
        random.shuffle(shuffle)
        
        data_clone = self.data.copy()
        labels_clone = self.labels.copy()
        
        for i, s in enumerate(shuffle):
            self.data[i] = data_clone[s]
            self.labels[i] = labels_clone[s]
            
        data_clone = None
        labels_clone = None
            
        # split data into train and validation sets
        split_idx = int(len(self.data)*(1-VAL_RATIO))
        self.val_data = self.data[split_idx:]
        self.val_labels = self.labels[split_idx:]
        self.data = self.data[:split_idx]
        self.labels = self.labels[:split_idx]
        
        # e.g. ["Sing", "me", "a", "song"]
        self.data_decoder = dict([(x[1], x[0]) for x in list(self.data_encoder.items())])  #Gives you word/genre from vector index
        # e.g. ["Rock", "Pop", "Hip Hop"]
        self.label_decoder = dict([(x[1], x[0]) for x in list(self.label_encoder.items())])

        self.num_classes = len(self.label_encoder)

        #print([data_enconder[word] for word in data[-1]])
        self.model = rnn(len(self.data_encoder) + 1, [128], [128], self.num_classes)
        
        self.best_acc = 0

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)

    def get_pred(self, lyrics):
        """Feed an entire song's lyrics into RNN word-by-word, recording the prediction for each word.
        Returns:
            preds (list): predicted labels for each word in lyrics.
        """
        rec = self.model.init_rec()
        preds = Variable(torch.FloatTensor(len(lyrics), self.model.n_classes).zero_())
        for i, word in enumerate(lyrics):
            #print(word)
            input = Variable(torch.zeros(1, len(self.data_encoder)+1))
            if word in self.data_encoder:
                input[0, self.data_encoder[word.lower()]] = 1
            else:
                input[0, 0] = 1

            pred, rec = self.model.forward(input, rec)
            preds[i] = pred

        return preds


    def get_accuracy(self):
        print("Validating model...")
        correct = 0
        for i, song in enumerate(self.val_data):
            pred = self.get_pred(song)
            value, index = torch.max(torch.mean(pred, 0, True), 1)
            print(self.label_encoder[self.val_labels[i].lower()])
            print(int(index))
            if self.label_encoder[self.val_labels[i].lower()] == int(index):
                correct += 100

        acc = correct/len(self.val_data)

        if acc > self.best_acc:
            self.best_acc = acc

        return acc

    def save_checkpoint(self, epoch, isbest=False, path=LOG_PATH):
        torch.save({
            'epoch':epoch,
            'state_dict':self.model.state_dict(),
            'best_acc':self.best_acc,
            'optimizer':self.optimizer.state_dict(),
            'data_encoder':self.data_encoder,
            'label_encoder':self.label_encoder
        }, path)
        if isbest:
            torch.save({
                'epoch':epoch,
                'state_dict':self.model.state_dict(),
                'best_acc':self.best_acc,
                'optimizer':self.optimizer.state_dict(),
                'data_encoder':self.data_encoder,
                'label_encoder':self.label_encoder
        }, SAVE_PATH)

    def train(self, start_epoch=0):
        """
        Params:
            start_epoch (int): current epoch number (non-zero if model was reloaded).
        """

        #for param in self.model.parameters():
        #    print(param.size())

        for i in range(start_epoch, MAX_EPOCHS):
            songs = [[self.data[i], self.labels[i]] for i in range(len(self.data))]
            random.shuffle(songs)
            preds = Variable(torch.FloatTensor(BATCH_SIZE, self.num_classes).zero_())
            labels = Variable(torch.LongTensor(BATCH_SIZE).zero_())
            for i, song in enumerate(songs):
                lyrics = song[0]
                genre = song[1]
                pred = self.get_pred(lyrics)
                labels = Variable(torch.LongTensor([self.label_encoder[genre.lower()] for i in pred]))

                preds = pred

                if (i+1)%PRINTERVAL == 0:
                    value, index = torch.max(torch.mean(pred, 0, True), 1)
                    #print(preds[i])
                    #print(labels[i])
                    print(str(i) + ": (guess) " + self.label_decoder[int(index[0])] + "=?=" + genre + " (actual)\tconfidence=" + str(float(value[0])))

                if (i+1)%BATCH_SIZE == 0 or i >= len(songs)-1:
                    self.loss = self.criterion(preds, labels)

                    self.optimizer.zero_grad()

                    self.loss.backward()

                    self.optimizer.step()
                    preds = Variable(torch.FloatTensor(BATCH_SIZE, self.num_classes).zero_())
                    labels = Variable(torch.LongTensor(BATCH_SIZE).zero_())
            # save checkpoint
                
                if (i+1)%SAVE_INTERVAL == 0:
                    acc = self.get_accuracy()
                    print("\nLoss:",self.loss.data[0],"\nAccuracy:", acc)
                    self.save_checkpoint(i+1, isbest=(acc == self.best_acc))
            
            acc = self.get_accuracy()
            print("\nLoss:",self.loss.data[0],"\nAccuracy:", acc)
            self.save_checkpoint(i+1, isbest=(acc == self.best_acc))

def main():
    trainer = Trainer()
    start_epoch = 0
    if os.path.isfile(LOG_PATH):
        print("Checkpoint found! Loading...")
        try:
            checkpoint = torch.load(LOG_PATH)
            start_epoch = checkpoint['epoch']
            trainer.model.load_state_dict(checkpoint['state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
            trainer.best_acc = checkpoint['best_acc']
            trainer.data_encoder = checkpoint['data_encoder']
            trainer.label_encoder = checkpoint['label_encoder']
            trainer.data_decoder = dict([(x[1], x[0]) for x in list(trainer.data_encoder.items())])  #Gives you word/genre from vector index
            trainer.label_decoder = dict([(x[1], x[0]) for x in list(trainer.label_encoder.items())])
        except RuntimeError as e:
            print("ERR: Failed to load checkpoint!")
            print(e)
    trainer.train(start_epoch=start_epoch)

if __name__ == "__main__":
    main()


