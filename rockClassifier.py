import os
from utils import dataset
import sys
import csv


# Simple Program that counts the number of Rock and Not Rock entries that are found in a CSV file
# The Genre of the song should always be listed at index 1 of the row
def main():
	# path = os.getcwd() + "/dataset/cleaned_lyrics.csv"
	path = os.getcwd() + "/dataset/nl_features_subset.csv"
	csv_reader = dataset.load_data(path)

	rock_count = 0
	not_rock_count = 0
	NA_count = 0
	other_count = 0

	for line in csv_reader:
		if len(line) == 0:
			continue
		if "genre" in line or "annotations" in line:
			continue

		if line[1] == "Rock":
			rock_count += 1
		elif line[1] == "Not Available":
			NA_count += 1
		elif line[1] == "Other":
			other_count += 1
		else:
			not_rock_count += 1

	print("Rock count is : " + str(rock_count))
	print("Not Rock Count is: " + str(not_rock_count))
	print("Not Available Count is " + str(NA_count))
	print("Other count is " + str(other_count))

if __name__ == "__main__":
	main()