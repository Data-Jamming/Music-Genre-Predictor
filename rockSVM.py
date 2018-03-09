from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from utils import dataset
import os
import random

# Read features from a CSV
'''
The order they will be found in the csv file:
1. Genre
2. Word Count
3. Verb Count
4. Noun COunt
5. Adjective Count
6. Determinant Count
7. Preposition Count
8. Adverb Count
9. Conjunction Count
10. Total Syllable Count
11. Average Syllable per Line
12. Some sort of encoding of rhyme scheme

'''
def main():
	path = os.getcwd() + '/dataset/nl_features.csv'
	csv_reader = dataset.load_data(path)
	matrix = []
	for line in csv_reader:
		if "song" in line or "genre" in line or len(line) == 0:
			# Ignore the head of the csv and any empty lines
			continue
		
		currentSong = []
		for i in range(1, len(line)):
			# check to see if we are working with the genre
			if i == 1:
				if line[i] == "Rock":
					# 1 is rock
					currentSong.append(1)
				else:
					# 0 is not rock
					currentSong.append(0)
			else:
				currentSong.append(float(line[i]))

		matrix.append(currentSong)
		# matrix.append([int(line[0]), float(line[1]), float(line[2]), float(line[3]), ])

	#print("Matrix before shuffling the contents")
	#print(matrix)
	#print("Matrix after shuffling the contents")
	#random.shuffle(matrix)
	#print(matrix)

	x = []
	y = []
	for line in matrix:
		# Alternatively I could also use dfferent permutations of the feature set
		# I probably need a better way of determining which bactch will be run at a time

		'''
		The different permutations of the dataset that can be ran
		0 = All Features
		1 = Only Parts of Speech - word count, verb count, noun count, adjective count, determinant count, preposition count
		2 = Syllable count and Average Syllable per line
		3 = Rhyme Scheme
		'''
		
		# permutation = 0

		# if permutation == 0:
		# 	x.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11])])
		# elif permutation == 1:
		# 	x.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8])])
		# elif permutaion == 2:
		# 	x.append([float(line[9]), float(line[10])])
		# elif permutation == 3:
		# 	x.append([float(line[11])])

		# Note: the name of the song is at 1, and everything else follows
		#x.append([float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12]), float(line[13]), float(line[14]), float(line[15]), float(line[16]), float(line[17]), float(line[18])])
		x.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])]
		y.append(int(line[0]))
	x = np.asarray(x)
	y = np.asarray(y)

	#print("The Data Matrix X: ")
	#print(x)
	#print("The Classification vector y: ")
	#print(y)
	x_scaled = preprocessing.scale(x)

	# Next I would need some subset of the data matrix and classes to be used for training and testing
	
	# Note: support vector machine algorithms are not scale invariant, so it is highly recommended to scale your data
	# For example, scale each attribute on the inmput vector X to [0,1] or [-1,1]
	# Note: the same scaling must be applied to the test vector to obtain meaningful results

	# Before doing any training or testing, we will first need to shuffle the data
	# and determine what portion is to be used for training / validation / testing

	# Test with different kernel functions, and different values for the slack variable
	
	# for k in ['linear', 'poly', 'sigmoid', 'rbf']:
	# 	clf = SCV(kernel = k)
	# 	clf.fit(x, y)
	# 	scr = clf.score(test_x, test_y)
	# 	print("\t",k, scr, clf.support_vectors_.shape)

	# Setting C: c is 1 by default, and it's a reasonable default choice. If you have a lot of noisy observations, you should decrease it
	# it corresponds to regularize more the estimation

	x_train = x[:80]
	x_test = x[80:]
	y_train = y[:80]
	y_test = y[80:]

	for k in ['linear', 'poly', 'sigmoid', 'rbf']:
	 	clf = SVC(kernel = k)
	 	clf.fit(x, y)
	 	scr = clf.score(x_test, y_test)
	 	print("\t",k, scr, clf.support_vectors_.shape)
	#print("The value of x_train is:", x_train)
	#print("The value of x_test is:", x_test)
	#for k in ['linear', 'poly', 'sigmoid', 'rbf']:
	#	for i in range(1,100):
	#		slack = float(i) / 10.0
	#		clf = SVC(C = slack, kernel = k)
	#		clf.fit(x_train, y_train)
	#		scr = clf.score(x_test, y_test)
	#		print("\t",k, slack, scr, clf.support_vectors_.shape)

	


if __name__ == "__main__":
	main()