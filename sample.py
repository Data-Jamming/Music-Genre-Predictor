# lib/framework imports here

# project imports here
from utils import dataset

def main():
    csv_reader = dataset.load_data("/Users/JC/code/data-mining-seng-474/project/neural-network-fun/dataset/lyrics.csv")
    feature_names = next(csv_reader)
    print(f"Found features={feature_names}")


if __name__ == "__main__":
    main()