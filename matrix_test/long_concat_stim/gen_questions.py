#!/usr/bin/env python3
import sys
sys.path.insert(0, "../helper_modules/")

from filesystem import globDir
import pdb
import csv
import random
from natsort import natsorted

def main():
    word_files = globDir("./out/stim/", "stim_words_*.csv")
    q_files = [x.replace("word", "question") for x in word_files]
    for word_file, q_file in zip(word_files, q_files):
        with open(word_file, 'r') as wordfile, open(q_file, 'w') as qfile:
            wordreader = csv.reader(wordfile)
            qwriter = csv.writer(qfile)
            sentences = [line for line in wordreader]
            n_sentences = len(sentences)
            q1_population = sentences[:n_sentences//2]
            q2_population = sentences[n_sentences//2:]
            q1 = random.choice(q1_population)
            q2 = random.choice(q2_population)

            blank_ind1 = random.randint(0, len(q1)-1)
            q1_ans = q1[blank_ind1]
            q1[blank_ind1] = '_'

            blank_ind2 = random.randint(0, len(q2)-1)
            q2_ans = q2[blank_ind2]
            q2[blank_ind2] = '_'
            q1 = [" ".join(q1), q1_ans]
            q2 = [" ".join(q2), q2_ans]

            qwriter.writerows([q1, q2])


if __name__ == "__main__":
    main()
