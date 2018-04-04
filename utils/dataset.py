import csv
import nltk

# arbitrary!
NUM_GENRES = 10

def load_data(dataset_path):
    # must specify newline='' to interpret newlines in quoted fields properly
    csv_file = open(dataset_path, encoding="utf8", newline='')
    csv_reader = csv.reader(csv_file)
    return csv_reader

def get_stratified_data(datasize):
    """Retrieves genre and lyrics for specified number of entries.
    NOTE: classes are balanced exactly, with datasize//NUM_GENRES entries each.

    Returns:
        data (list): lyrics, tokenized. e.g. [["I", "want", "you", "so", "bad"]]
        labels (list): genres e.g. ["Rock"]
    """

    csv_reader = load_data("dataset/cleaned_lyrics.csv")
    # skip header names (e.g. "genre", "lyrics", etc.)
    next(csv_reader)

    data, labels = [], []
    genre_count = dict()
    max_per_genre = datasize // NUM_GENRES
    # if datasize is divisible by NUM_GENRES, max_entries = datasize
    max_entries = NUM_GENRES * max_per_genre

    num_entries = 0
    while True:
        try:
            song = next(csv_reader)
        except StopIteration as e:
            print("Reached end of dataset")
            break

        # based on our running genre count, determine whether
        # we want to keep this entry
        genre = song[1]
        cur_genre_count = genre_count.get(genre)
        if cur_genre_count is None:
            # add genre to count if haven't seen it before
            genre_count[genre] = 1
        elif cur_genre_count >= max_per_genre:
            # skip to next entry if we already have enough of this genre
            continue
        else:
            genre_count[genre] += 1

        num_entries += 1

        data.append(nltk.word_tokenize(song[2]))
        labels.append(genre)

        # break out of loop if we've reached our total
        if num_entries >= max_entries:
            break
    return data, labels

def read_first_n_entries(n=1000):
    data, labels = [], []
    for i in range(datasize):
        try:
            song = next(csv_reader)
            data.append(nltk.word_tokenize(song[2]))
            labels.append(song[1])
        except StopIteration:
            break;
    return data, labels

