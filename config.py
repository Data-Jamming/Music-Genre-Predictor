import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Absolute path to your csv dataset file",
    default=f"{os.getcwd()}/dataset/lyrics.csv",
)

def get_config():
    return parser.parse_args()