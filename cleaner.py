# lib/framework imports here

import os
# project imports here
from utils import dataset
import langid
import sys
import re
import csv

def clean(row):
    if(row[1]==None or row[4]==None or row[5]==None):
        return None
    genre = no_genre(row[4])
    if(genre is None):
        return None
    split = row[5].split('\n')
    if(len(split) < 5):
        return None
    l= is_en(row[5],split)
    if(l==False):
        return None
    rmv_annotation(split)
    return "\n".join(split);


# Only checking for the "Not Available" case, decide if we want to remove "Other" as well
def no_genre(genre):
    if genre == "Not Available":
        return None
    else:
        return genre

def is_en(text,split):
    not_eng=0;
    for line in split:
        l= langid.classify(line)
        if(l[0]!='en'):
            not_eng = not_eng+1
    if(langid.classify(text)[0] != 'en' or not_eng > (len(split)/8)):
        return False
    return True

def rmv_annotation(split):
    p=re.compile("(\(x[1-9]\))" , re.IGNORECASE)
    p2=re.compile("(\([1-9]x\))" , re.IGNORECASE)
    for i in range(len(split)):
        arr = split[i].split()
        for word in arr:
            m = p.match(word)
            m2= p2.match(word)
            if(m!=None or m2!=None):
                st = m.group() if m!=None else m2.group()
                reps =int(m.group()[2] if m!=None else m2.group()[1])
                place = arr.index(word);
                if(len(arr)==1):
                    replacement = (split[i-1]) * reps
                    split[i] = replacement;
                elif(place==0):
                    replacement = ((" ".join(arr[1:])) + '\n') * reps
                    replacement = replacement[:-1]
                    split[i]= replacement
                else:
                    replacement = ((" ".join(arr[0:place])) + '\n') * reps
                    replacement = replacement[:-1]
                    split[i]= replacement

def main():
    path = os.getcwd() + '/dataset/lyrics.csv'
    new_path = os.getcwd() + '/dataset/cleaned_lyrics.csv'
    #need to check for cache?
    csv_reader = dataset.load_data(path)
    f=open(new_path, 'w', encoding="utf8")
    fieldnames = ['song', 'genre','lyrics']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for line in csv_reader:
        res = clean(line)
        if(res!=None):
            writer.writerow({'song': line[1], 'genre': line[4], 'lyrics':res})
        
main()
