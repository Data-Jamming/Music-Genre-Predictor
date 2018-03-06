# lib/framework imports here
from config import get_config

# project imports here
from utils import dataset

import nltk

def main(config):
    csv_reader = dataset.load_data(config.dataset_path)
    feature_names = next(csv_reader)
    print(f"Found features={feature_names}")

if __name__ == "__main__":
    config = get_config()
    main(config)
