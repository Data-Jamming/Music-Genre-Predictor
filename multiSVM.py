from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from utils import dataset
import os
import random
import math
from collections import Counter
from sklearn.externals import joblib

genres_list = ["Rock", "Country", "Electronic", "Folk", "Hip-Hop", "Indie", "Jazz", "Metal", "Pop", "R&B"]

x_train = None
x_test = None

y_train = None
y_test = None

training_size = None
testing_size = None
overall_size = None

x_scaled = None

matrix = None

# Return the genre that is matched to a specific integer
def int_to_genre(i):
	if i == 1:
		return "Rock"
	elif i == 2:
		return "Country"
	elif i == 3:
		return "Electronic"
	elif i == 4:
		return "Folk"
	elif i == 5:
		return "Hip-Hop"
	elif i == 6:
		return "Indie"
	elif i == 7:
		return "Jazz"
	elif i == 8:
		return "Metal"
	elif i == 9:
		return "Pop"
	elif i == 10:
		return "R&B"


# For each category that we predicted incorrecty, what was the prediction that we actually made for that category
# ex. Correct Category = ROCK, in the list number of times that we predicted Country, Pop, etc...

# This will be useful for something like Country which could appear in the testing set ex. 2500 times but we actually only predict it 50 times, are we predicting ROCK all of the other times
# Hence, this will tell us for each category that we are guessing wrong, what are we likely to predict instead
def incorrect_predictions(preds, y_test):
	global genres_list

	# Dictionary of lists
	incorrect = {"Rock": [], "Country": [], "Electronic": [], "Folk": [], "Hip-Hop": [], "Indie": [], "Jazz": [], "Metal": [], "Pop": [], "R&B": []}
	for i in range(len(preds)):
		if preds[i] != y_test[i]:
			correct_genre = int_to_genre(y_test[i])
			incorrect[correct_genre].append(int_to_genre(preds[i]))

	# Iterate through all of the examples and print out what they were
	print("For each Genre that was misclassified, the genre that was predicted by our SVM")
	for genre in genres_list:
		print(genre + ":", Counter(incorrect[genre]))
	print()


# Returns a dictionary of the number of times that a particular genre appeared in a particular array - could either be our prediction matrix, or the actual classification matrix
def genre_count(y_test):
	counts = {"Rock": 0, "Country": 0, "Electronic": 0, "Folk": 0, "Hip-Hop": 0, "Indie": 0, "Jazz": 0, "Metal": 0, "Pop": 0, "R&B": 0}

	for i in range(len(y_test)):
		if y_test[i] == 1:
			counts["Rock"] += 1
		elif y_test[i] == 2:
			counts["Country"] += 1
		elif y_test[i] == 3:
			counts["Electronic"] += 1
		elif y_test[i] == 4:
			counts["Folk"] += 1
		elif y_test[i] == 5:
			counts["Hip-Hop"] += 1
		elif y_test[i] == 6:
			counts["Indie"] += 1
		elif y_test[i] == 7:
			counts["Jazz"] += 1
		elif y_test[i] == 8:
			counts["Metal"] += 1
		elif y_test[i] == 9:
			counts["Pop"] += 1
		elif y_test[i] == 10:
			counts["R&B"] += 1

	return counts



# Of the number of times that we predicted a particular category, how many times was the prediction of that category actually correct
def genres_correct(preds, y_test):

	rock_correct = 0
	country_correct = 0
	electronic_correct = 0
	folk_correct = 0
	hiphop_correct = 0
	indie_correct = 0
	jazz_correct = 0
	metal_correct = 0
	pop_correct = 0
	rb_correct = 0

	# just create a dictionary of these
	for i in range(len(preds)):
		if preds[i] == y_test[i] and preds[i] == 1:
			rock_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 2:
			country_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 3:
			electronic_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 4:
			folk_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 5:
			hiphop_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 6:
			indie_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 7:
			jazz_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 8:
			metal_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 9:
			pop_correct += 1
		elif preds[i] == y_test[i] and preds[i] == 10:
			rb_correct += 1

	# Retrieve dictionary of count of each genre
	genre_counts = genre_count(y_test)

	rock_accuracy = rock_correct/genre_counts["Rock"] * 100 if genre_counts["Rock"] > 0 else 0
	country_accuracy = country_correct/genre_counts["Country"] * 10 if genre_counts["Country"] > 0 else 0
	electronic_accuracy = electronic_correct/genre_counts["Electronic"] * 100 if genre_counts["Electronic"] > 0 else 0
	folk_accuracy = folk_correct/genre_counts["Folk"] * 100 if genre_counts["Folk"] > 0 else 0
	hiphop_accuracy = hiphop_correct/genre_counts["Hip-Hop"] * 100 if genre_counts["Hip-Hop"] > 0 else 0
	indie_accuracy = indie_correct/genre_counts["Indie"] * 100 if genre_counts["Indie"] > 0 else 0
	jazz_accuracy = jazz_correct/genre_counts["Jazz"] * 100 if genre_counts["Jazz"] > 0 else 0
	metal_accuracy = metal_correct/genre_counts["Metal"] * 100 if genre_counts["Metal"] > 0 else 0
	pop_accuracy = pop_correct/genre_counts["Pop"] * 100 if genre_counts["Pop"] > 0 else 0
	rb_accuracy = rb_correct/genre_counts["R&B"] * 100 if genre_counts["R&B"] > 0 else 0

	print("1. Rock correct:", rock_correct, "- Rock incorrect:", (genre_counts["Rock"] - rock_correct), "- Accuracy:", rock_accuracy)
	print("2. Country correct:", country_correct, "- Country incorrect:", (genre_counts["Country"] - country_correct), "- Accuracy:", country_accuracy)
	print("3. Electronic correct:", electronic_correct, "- Electronic incorrect:", (genre_counts["Electronic"] - electronic_correct), "- Accuracy:", electronic_accuracy)
	print("4. Folk correct:", folk_correct, "- Folk incorrect:", (genre_counts["Folk"] - folk_correct), "- Accuracy:", folk_accuracy)
	print("5. Hip Hop correct:", hiphop_correct, "- Hip-Hop incorrect:", (genre_counts["Hip-Hop"] - hiphop_correct), "- Accuracy:", hiphop_accuracy)
	print("6. Indie correct:", indie_correct, "- Indie incorrect:", (genre_counts["Indie"] - indie_correct), "- Accuracy:", indie_accuracy)
	print("7. Jazz correct:", jazz_correct, "- Jazz incorrect:", (genre_counts["Jazz"] - jazz_correct), "- Accuracy:", jazz_accuracy)
	print("8. Metal correct:", metal_correct, "- Metal incorrect:", (genre_counts["Metal"] - metal_correct), "- Accuracy:", metal_accuracy)
	print("9. Pop correct:", pop_correct, "- Pop incorrect:", (genre_counts["Pop"] - pop_correct), "- Accuracy:", pop_accuracy)
	print("10. R&B Correct:", rb_correct, "- R&b incorrect:", (genre_counts["R&B"] - rb_correct), "- Accuracy:", rb_accuracy)
	print()



'''
Trains and tests a SVM with varying Kernels and Slack Variable Coefficients

'''
def automated_test():
	global x_train
	global x_test
	global y_train
	global y_test
	global x_scaled
	global training_size
	global testing_size
	print()
	overall_best_accuracy = 0
	overall_best_kernel = None
	overall_best_slack_variable = None
	preds=[]
	for k in ['linear', 'poly', 'rbf']:
	# for k in ['linear']:
		# Create a list and store the highest one each time
		max_accuracy = 0
		max_slack_variable = None

		# for i in [0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.3, 1.5, 1.7, 2, 2.3, 2.5, 2.7, 3, 3.3, 3.5, 3.7, 3.9, 4, 10, 15, 20, 30, 50, 70, 100, 200, 300, 400, 500]:
		for i in [1, 50, 100]:
			clf = SVC(C = i, cache_size = 1000, kernel = k, decision_function_shape = "ovo")
			clf.fit(x_train, y_train)
			for j in range(len(x_test)):
				next = j + 1
				preds.append(clf.predict(x_test[j: next]).item(0))
			scr = clf.score(x_test, y_test)

			if scr >= max_accuracy:
				max_accuracy = scr
				max_slack_variable = i

			if scr >= overall_best_accuracy:
				overall_best_accuracy = scr
				overall_best_kernel = k
				overall_best_slack_variable = i
			print("actual distribution")

			# print y_test in a readable format
			y_test_labels = []
			for a in range(len(y_test)):
				y_test_labels.append(int_to_genre(y_test[a]))

			print(Counter(y_test_labels))
			print()
			print("predicted distribution")

			# Print preds in a readable format
			preds_labels = []
			for b in range(len(preds)):
				preds_labels.append(int_to_genre(preds[b]))

			print(Counter(preds_labels))
			# bad_count =0
			# for l in range(len(preds)):
			# 	if(preds[l]!=y_test[l]):
			# 		bad_count = bad_count +1
			# print(bad_count)
			genres_correct(preds, y_test)
			incorrect_predictions(preds, y_test)
			preds.clear()
			print("Kernel:", k , "Slack variable:", i, "Score:", scr)
			print("***********************************\n")
		print()
		print("*********************************")
		print("Current Kernel:", k)
		print("Max Accuracy:", max_accuracy)
		print("Slack Variable for Max Accuracy:", max_slack_variable)
		print("*********************************")
		print()

	print("***************************************")
	print("OVERALL STATISTICS:")
	print("Overall best accuracy was:", overall_best_accuracy)
	print("Overall best accuracy occurred with kernel:", overall_best_kernel, "and Slack Variable", overall_best_slack_variable)
	print("***************************************")

	rock_count = 0
	for i in y_test:
		if i == 1:
			rock_count += 1

	accuracy = rock_count / len(y_test)
	print("Total number of Rock songs found in test set: " + str(rock_count))
	print("Total number of songs found in the test set: " + str(len(y_test)))
	print("Accuracy of a Naive Classifier that always picks rock: " + str(accuracy))


'''
Train and test a single SVM according to specific parameters
Can also print out the generated classes of the testing data and
compare it with the values in the classification vector
'''
def manual_test():
	global x_train
	global x_test
	global y_train
	global y_test
	global x_scaled
	global training_size
	global testing_size
	global matrix

	clf = SVC(C = 100, cache_size = 1000, kernel = "linear", decision_function_shape = "ovo") # both linear and rbf appear to give an accuracy around 80%, decreasing the Slack Variable decreases accuracy
	clf.fit(x_train, y_train)

	# Once we have the model saved we can just import the model instead of having to train the model
	# Step 1: dump the model after "fitting" it with the above code
	# Step 2: Remove the "fit" command and just "load" the model
	# joblib.dump(clf, "model1.pkl")
	# clf = joblib.load("model1.pkl")

	# Individually test each of the points in the testing set
	answers = []
	for i in range(training_size,len(x_scaled)):
		next = i+1
		answer = clf.predict(x_scaled[i: next])
		answers.append(answer[0])

	genre_count(y_test)
	incorrect_predictions(answers, y_test)

	print("Answers generated by the SVM, length =", len(answers))
	print(answers)

	print("Answers defined by the dataset, length = ", len(y_test))
	print(y_test)

	retrieve_song_names(matrix)

	correct = 0
	random_correct = 0
	test_rock_count = 0
	for i in range(0, len(y_test)):
		if answers[i] == y_test[i]:
			correct += 1
		if y_test[i] == 1:
			# incrememnt random_correct
			random_correct += 1
		if y_test[i] == 1:
			test_rock_count += 1


	print("The actual number of rock songs found in the testing set", test_rock_count)

	print("My Calculated percentage: " + str(correct/len(y_test)))
	scr = clf.score(x_test, y_test)
	print("The percentage calculated by sci-kit Learn: " + str(scr))
	print("The accuracy of a random classifer that always predicts Rock", ((random_correct/len(y_test) * 100)))


# Retrieve the name of the songs in the given matrix
def retrieve_song_names(matrix):
	global training_size
	global overall_size
	song_names = []
	# Some predefined range based on the subset that I was testing on - this was the testing subset in this case
	for i in range(training_size,overall_size):
		line = matrix[i]
		song_names.append(line[0])
	print(song_names)


# Read from the "csv_reader" and generate a matrix
def generate_full_matrix(csv_reader):
	matrix = []
	for line in csv_reader:
		if "song" in line or "genre" in line or len(line) == 0:
			# Ignore the head of the csv and any empty lines
			continue
		currentSong = []
		for i in range(0, len(line) -1):
			# check to see if we are working with the genre
			if i == 0:
				# Include the name of the song
				currentSong.append(line[0])
			elif i == 1:
				if line[i] == "Rock":
					# 1 is rock
					currentSong.append(1)
				elif line[i] == "Country":
					# 2 is Country
					currentSong.append(2)
				elif line[i] == "Electronic":
					# 3 is Electronic
					currentSong.append(3)
				elif line[i] == "Folk":
					# 4 is Folk
					currentSong.append(4)
				elif line[i] == "Hip-Hop":
					# 5 is Hip-Hop
					currentSong.append(5)
				elif line[i] == "Indie":
					# 6 is Indie
					currentSong.append(6)
				elif line[i] == "Jazz":
					# 7 is Jazz
					currentSong.append(7)
				elif line[i] == "Metal":
					# 8 is Metal
					currentSong.append(8)
				elif line[i] == "Pop":
					# 9 is Pop
					currentSong.append(9)
				elif line[i] == "R&B":
					# 10 is R&B
					currentSong.append(10)
				else:
					# 0 is not rock - this case should never appear
					currentSong.append(0)
			else:
				currentSong.append(float(line[i]))

		matrix.append(currentSong)

		# Shuffle the matrix before separating into the feature matrix and the classification vector
		random.shuffle(matrix)
	return matrix


def main():
	global x_train
	global x_test
	global y_train
	global y_test
	global x_scaled
	global training_size
	global testing_size
	global matrix
	global overall_size

	# path = os.getcwd() + '/dataset/nl_features_subset.csv'
	path = os.getcwd() + '/dataset/nl_features.csv'
	csv_reader = dataset.load_data(path)
	matrix = generate_full_matrix(csv_reader)

	x = []
	y = []
	# Separate the matrix into the feature matrix and the classification vector
	for line in matrix:

		'''
		Ordering found in the CSV - song name has already been removed at this time, and genre is either Rock (1) or Not rock (0)
		0. Song Name - include song name so that we can go backwards
		1. genre
		2. annotations
		3. syllables
		4. syll_per_line
		5. verb
		6. adj
		7. noun
		8. pre-det
		9. det
		10. prep
		11. pronoun
		12. pos
		13. conj
		14. cardinal
		15. adverb
		16. particle
		17. exist
		18. inj
		19. aux
		20. aa - rhyme scheme
		21. abab - rhyme scheme
		22. abba - rhyme scheme
		'''

		#TODO: Allow for different permutations of above to be ran, if we have to retrain the model each time
		# we can just manually modify this

		#Include all of the available features
		x.append([float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]), float(line[12]), float(line[13]), float(line[14]), float(line[15]), float(line[16]), float(line[17]), float(line[18]), float(line[19]), float(line[20]), float(line[21]), float(line[22])])


		y.append(int(line[1]))
	x = np.asarray(x)
	y = np.asarray(y)

	# Scale the data matrix
	# Note: support vector machine algorithms are not scale invariant, so it is highly recommended to scale your data
	# For example, scale each attribute on the inmput vector X to [0,1] or [-1,1]
	x_scaled = preprocessing.scale(x)

	overall_size = len(x_scaled)

	# Training data should be 0 - training_size
	training_size = math.ceil(len(x_scaled) * .80)

	# Testing data should be testing_size (training_size + 1) - len(matrix)
	# I guess because of the inclusive, exclusive testing_size should just start where training_size ends
	testing_size = training_size

	# to be used for training - x_train should be used with y_train (they should correspond with eachother)
	x_train = x_scaled[:training_size]
	y_train = y[:training_size]

	# to be used for testing - x_test should be used with y_test (they should correspond with eachother)
	x_test = x_scaled[training_size:]
	y_test = y[training_size:]

	# manual_test()

	automated_test()

if __name__ == "__main__":
	main()
