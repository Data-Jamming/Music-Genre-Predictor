import os
from utils import dataset
import sys
import csv


# Simple Program that counts the number of Rock and Not Rock entries that are found in a CSV file
# The Genre of the song should always be listed at index 1 of the row
def main():
	# path = os.getcwd() + "/dataset/cleaned_lyrics.csv"
	path = os.getcwd() + "/dataset/nl_features.csv"
	csv_reader = dataset.load_data(path)

	rock_count = 0
	not_rock_count = 0
	NA_count = 0
	other_count = 0
	total_count = 0

	country_count = 0
	electronic_count = 0
	folk_count = 0
	indie_count = 0
	jazz_count = 0
	metal_count = 0
	pop_count = 0
	rb_count = 0
	hiphop_count = 0

	for line in csv_reader:
		if len(line) == 0:
			continue
		if "genre" in line or "annotations" in line:
			continue

		total_count += 1
		if line[1] == "Rock":
			rock_count += 1
		elif line[1] == "Country":
			country_count += 1
		elif line[1] == "Electronic":
			electronic_count += 1
		elif line[1] == "Hip-Hop":
			hiphop_count += 1
		elif line[1] == "Folk":
			folk_count += 1
		elif line[1] == "Indie":
			indie_count += 1
		elif line[1] == "Jazz":
			jazz_count += 1
		elif line[1] == "Metal":
			metal_count += 1
		elif line[1] == "Pop":
			pop_count += 1
		elif line[1] == "R&B":	
			rb_count += 1
		elif line[1] == "Not Available":
			NA_count += 1
		elif line[1] == "Other":
			other_count += 1
		else:
			not_rock_count += 1

	print("Rock count is : " + str(rock_count) + "\toverall % " + str(100* rock_count / total_count))
	print("Country count is : " + str(country_count) + "\toverall % " + str(100* country_count / total_count))
	print("Electronic count is : " + str(electronic_count) + "\toverall % " + str(100* electronic_count / total_count))
	print("Folk count is : " + str(folk_count) + "\toverall % " + str(100* folk_count / total_count))
	print("Hip-Hop count is: " + str(hiphop_count) + "\toverall % " + str(100* hiphop_count / total_count))
	print("Indie count is : " + str(indie_count) + "\toverall % " + str(100* indie_count / total_count))
	print("Jazz count is : " + str(jazz_count) + "\toverall % " + str(100* jazz_count / total_count))
	print("Metal count is : " + str(metal_count) + "\toverall % " + str(100* metal_count / total_count))
	print("Pop count is : " + str(pop_count) + "\toverall % " + str(100* pop_count / total_count))
	print("R&B count is : " + str(rb_count) + "\toverall % " + str(100* rb_count / total_count))
	print()
	print("Total number of songs: " + str(total_count))
	#print("Not Rock Count is: " + str(not_rock_count))
	#print("Not Available Count is " + str(NA_count))
	#print("Other count is " + str(other_count))

if __name__ == "__main__":
	main()