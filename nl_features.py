
import os
# project imports here
from utils import dataset
import sys
import csv
import string
import re
from nltk.corpus import cmudict
from nltk.corpus import stopwords
import nltk #change this later to a more specific import

#globals
dict = cmudict.dict()
vowels=['AO','AA','IY','UW','EH','IH','UH','AH','AX','AY','AE','EY','OW','AW','OY']
POS_counts= {'VB':0, 'JJ':0, 'WR':0, 'NN':0, 'DT':0, 'WD':0, 'IN':0, 'CC':0, 'CD':0, 'RB':0,'RP':0, 'UH': 0, 'EX':0, 'MD':0, 'PR':0, 'WP':0, 'PO':0, 'TO':0, 'PD':0}
annotation_count=0
avg_syll_per_line=0
total_sylls=0
word_count=0
aa_count=0
abab_count =0
abba_count =0



#is digit
def my_isdigit(s):
	 return all('0'<=c<='9' for c in s)

def count_syllables(word):
 syll_sum=0
 for z in word.split("_"):
  try:
   syll_sum= syll_sum + max([len(list(y for y in x if my_isdigit(y[-1]))) for x in dict[z.lower()]])
  except KeyError:
   return 0;
 return syll_sum

def count_things(sentences):
	global avg_syll_per_line
	global total_sylls
	global word_count
	global POS_counts

	avg_syll_per_line = 0
	total_sylls = 0
	word_count = 0

	# Reset all the key-value pairs in the dict
	for key in POS_counts:
		POS_counts[key] = 0

	for s in sentences:
		words = s.split()
		if(len(words)==0):
			continue
		word_count = word_count + len(words)
		tokens = nltk.pos_tag(words)
		for word, token in zip(words, tokens):
			num = count_syllables(word)
			if(num!=-1):
				total_sylls = total_sylls +num
				curr=token[1][:2]
				try:
					POS_counts[curr] = POS_counts[curr] +1
				except KeyError:
					continue
		avg_syll_per_line = total_sylls/len(s)

def strip_annotation(text):
	 global annotation_count
	 annotation_count = 0
	 p=re.compile("\[[a-zA-z0-9\s]+(:)?\]" , re.IGNORECASE)
	 p2=re.compile("Chorus(:?)" , re.IGNORECASE)
	 res = re.subn(p,"",text)
	 text = res[0]
	 annotation_count = annotation_count + res[1]
	 res = re.subn(p2,"",text)
	 text = res[0]
	 annotation_count = annotation_count + res[1]
	 regex = re.compile('[%s]' % re.escape(string.punctuation))
	 return regex.sub('', text)

def get_phonetic(word):
  try:
   result= dict[word][0]
  except KeyError:
   return -1
  return result

def is_sound_match(phon_1, phon_2):
	if(phon_1==-1 or phon_2==-1):
		return None
	has_vowel=False
	syll_match=0
	for x in range((len(phon_1) if len(phon_1) < len(phon_2) else len(phon_2))):
	   if(phon_1[len(phon_1)-1-x] != phon_2[len(phon_2)-1-x]):
	    break;
	   syll_match=syll_match+1
	if(syll_match==0):
		return None
	match = phon_1[len(phon_1)-syll_match:]
	for sound in match:
		if(sound[:-1] in vowels):
			return match
	return None

def get_rhyme_scheme(s):
    sound_dict ={}
    labels=[]
    letter = 65
    counter=0
    for line in s:
        tmp=65
        if(line==[]):
            continue
        if(labels==[]):
            sound_dict[chr(tmp)]=get_phonetic(line[len(line)-1])
            labels.append(chr(tmp))
            continue
        while(True):
            if(tmp > letter):
                letter =letter+1
                if(letter==90):
                  letter = 97
                sound_dict[chr(letter)] = get_phonetic(line[len(line)-1])
                break
            match = is_sound_match(get_phonetic(line[len(line)-1]), sound_dict[chr(tmp)])
            if(match!=None):
                sound_dict[chr(tmp)]= match
                break
            else:
                tmp = tmp + 1
                if(tmp == 90):
                   tmp=97
        labels.append(chr(tmp))
        counter = counter +1
    return "".join(['None' if elem is None else elem for elem in labels])

def count_rhymes(scheme):
	global aa_count
	global abab_count
	global abba_count
	aa_count =0
	abba_count=0
	abab_count=0
	for i in range(len(scheme)-1):
		if scheme[i]==scheme[i+1]:
			aa_count = aa_count +1
	for i in range(len(scheme)-3):
		if((scheme[i]==scheme[i+2]) and (scheme[i+1]==scheme[i+3])):
			abab_count = abab_count +1
		if((scheme[i]==scheme[i+3]) and (scheme[i+1]==scheme[i+2])):
			abba_count = abba_count +1

def highest_freq(sentences):
	words=[]
	for s in sentences:
		words += s.split()
	words= [w for w in words if w not in stopwords.words("english")]
	freqs = {}
	for w in words:
		try:
			freqs[w] = freqs[w] + 1
		except KeyError:
			freqs[w] =1
	return (max(freqs.values())/len(words)) if len(freqs.values())!=0 else 0

def gen_dict(lyrics, dict_res):
	global POS_counts
	global annotation_count
	global avg_syll_per_line
	global total_sylls
	global word_count
	global aa_count
	global abab_count
	global abba_count
	lyrics = strip_annotation(lyrics)
	sentences = lyrics.split("\n")
	sentences= [x for x in sentences if x!=[]]
	count_things(sentences)
	freq = highest_freq(sentences)
	splits=[]
	for s in sentences:
	    splits.append(s.split())
	splits= [x for x in splits if x!=[]]
	count_rhymes(get_rhyme_scheme(splits))
	dict_res['annotations'] = annotation_count
	dict_res['syllables'] = total_sylls
	dict_res['syll_per_line'] = avg_syll_per_line
	dict_res['words']= word_count
	dict_res['verb'] = POS_counts['VB']
	dict_res['adj'] = POS_counts['JJ'] + POS_counts['WR']
	dict_res['noun'] = POS_counts['NN']
	dict_res['pre-det'] = POS_counts['PD']
	dict_res['det'] = POS_counts['DT'] + POS_counts['WD']
	dict_res['prep'] = POS_counts['IN'] + POS_counts['TO']
	dict_res['pronoun'] = POS_counts['PR'] + POS_counts['WP']
	dict_res['pos'] = POS_counts['PO']
	dict_res['conj'] = POS_counts['CC']
	dict_res['adverb'] = POS_counts['RB']
	dict_res['particle'] = POS_counts['RP']
	dict_res['exist'] = POS_counts['EX']
	dict_res['inj'] = POS_counts['UH']
	dict_res['aux'] = POS_counts['MD']
	dict_res['cardinal'] = POS_counts['CD']
	dict_res['aa'] = aa_count
	dict_res['abab'] = abab_count
	dict_res['abba'] = abba_count
	dict_res['freq'] = freq


def main() :

	path = os.getcwd() + '/dataset/cleaned_lyrics.csv'
	csv_reader = dataset.load_data(path)
	new_path = os.getcwd() + '/dataset/nl_features_stratified2.csv'
	f=open(new_path, 'w', encoding="utf8")
	fieldnames = ['song', 'genre','annotations', 'syllables','syll_per_line','words','verb','adj','noun','pre-det','det','prep','pronoun','pos','conj','cardinal','adverb','particle','exist','inj','aux', 'aa','abab','abba','freq']
	writer = csv.DictWriter(f, fieldnames=fieldnames)
	writer.writeheader()
	counter=0
	data, labels = dataset.get_stratified_data(2000, True, False)

	# for reading from the stratified dataset
	for i in range(len(data)):
		res_dict={}
		res_dict['song'] = "Test"
		res_dict['genre'] = labels[i]
		gen_dict(data[i], res_dict)
	

		#if counter == 10001:
			#break
		writer.writerow(res_dict)
		counter = counter +1

	'''
	for line in csv_reader:
		res_dict={}
		res_dict['song'] = line[0]
		res_dict['genre'] = line[1]
		gen_dict(line[2], res_dict)
		if(counter==0):
			counter = counter +1
			continue

		if counter == 10001:
			break
		writer.writerow(res_dict)
		counter = counter +1
	'''
main()
