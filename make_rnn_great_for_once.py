from torch.nn import RNN

# default (arbitrary)
HIDDEN_SIZE=5
NUM_LAYERS=3
ACTIV="relu"

class MyRNN():
    def __init__(self, num_words, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, activ=ACTIV):
        my_rnn = RNN(
            input_size=num_words,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=activ,
        )

def main():
    print("Not doing much in this 'main' function...")
    my_rnn = MyRNN(10)

if __name__ == "__main__":
    main()