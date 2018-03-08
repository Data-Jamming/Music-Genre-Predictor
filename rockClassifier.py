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
	other_count = 0

	for line in csv_reader:
		if line[1] == "Rock":
			rock_count += 1
		elif line[1] == "Not Available":
			NA_count += 1
		elif line[1] == "Other":
			other_count += 1
		else:
			not_rock_count += 1

		# Perform Natural Language Processing with the lyrics -> line[2]

	print("Rock count is : " + str(rock_count))
	print("Not Rock Count is: " + str(not_rock_count))
	print("Not Available Count is " + str(NA_count))
	print("Other count is " + str(other_count))

	new_path = os.getcwd() + '/dataset/rock_features.csv'

	f = open(new_path, 'w', encoding='utf8')
	fieldnames = ['genre', 'wordCount', 'verbCount', 'adjCount', 'determCount', 'prepCount', 'adverbCount', 'conjunctionCount', 'totSyllableCount', 'avgSyllablePerLine']
	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()
	writer.writerow({'genre': 1, 'wordCount': 1, 'verbCount': 1, 'adjCount': 1, 'determCount': 1, 'prepCount': 1, 'adverbCount': 1, 'conjunctionCount': 1, 'totSyllableCount': 1, 'avgSyllablePerLine': 1})


if __name__ == "__main__":
	main()