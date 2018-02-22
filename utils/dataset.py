import csv

def load_data(dataset_path):
    # must specify newline='' to interpret newlines in quoted fields properly
    csv_file = open(dataset_path, newline='')
    csv_reader = csv.reader(csv_file)
    return csv_reader