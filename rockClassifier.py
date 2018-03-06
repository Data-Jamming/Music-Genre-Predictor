import os
from utils import dataset
import sys
import csv


# Process the data to remove the "Not Available" Genres - not complete yet
def main():
	path = os.getcwd() + "/dataset/cleaned_lyrics.csv"
	csv_reader = dataset.load_data(path)

	rock_count = 0
	not_rock_count = 0
	NA_count = 0

	for line in csv_reader:
		if line[1] == "Rock":
			rock_count += 1
		elif line[1] == "Not Available":
			NA_count += 1
		else:
			not_rock_count += 1

	print("Rock count is : " + str(rock_count))
	print("Not Rock Count is: " + str(not_rock_count))
	print("Not Available Count is " + str(NA_count))

if __name__ == "__main__":
	main()