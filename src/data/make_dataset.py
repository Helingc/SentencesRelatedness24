import nltk
from nltk.corpus import wordnet
import random
import pandas as pd
import re

#nltk.download('wordnet')

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def get_replacable_word(seq,sep = '[SEP]'):
    seq = seq.replace(sep,'')
    seq = re.sub(r"[^a-zA-Z0-9]+", ' ', seq)
    seq = list(set(seq.split()))
    candidates = [cand for cand in seq if get_synonyms(cand) != []]
    return candidates[random.sample(range(5),1)[0]]




train_data = pd.read_csv('/Users/lemarx/Documents/01_projects/SentencesRelatedness24/data/raw/eng_train.csv')

sep = '[SEP]'
train_data['input'] = train_data.apply(lambda row : row['Text'].replace('\n',sep),axis = 1)

print(get_replacable_word(train_data['input'].loc[0]))
#print(train_data['input'].loc[0].replace(get_replacable_word(train_data['input'].loc[0]),get_synonyms(get_replacable_word)))
print(get_synonyms('plug')[0])