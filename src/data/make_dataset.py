import nltk
from nltk.corpus import wordnet as wn
import random
import pandas as pd
import re
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import lightning as pl
from sklearn.model_selection import train_test_split
import string





class str_dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, syn_replace = False, change_random_letter = False, sep = '[SEP]'):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.syn_replace = syn_replace
        self.change_random_letter = change_random_letter
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')

        self.dataframe['input'] = self.dataframe.apply(lambda row : row['Text'].replace('\n',sep),axis = 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the features and target from the DataFrame
        # Adjust this based on your DataFrame structure
        features = self.dataframe['input'].loc[idx]

        if self.change_random_letter:
            features = self.apply_change_letter(features)

        if self.syn_replace and random.random() < 0.3:
            features = self.apply_syn_replacement(features)

        token = self.tokenizer(features, return_tensors='pt', truncation='longest_first', padding='max_length',max_length=265)

        input_ids = token['input_ids'].squeeze()
        attention_mask = token['attention_mask'].squeeze()

        score = self.dataframe['Score'].loc[idx]

        score = torch.tensor(score, dtype=torch.float32).unsqueeze(dim=0)  # Adjust the dtype as needed

        return input_ids, attention_mask, score

    def get_synonym(word : str, pos : str) -> str:
        synonyms = []
        if pos == None:
            return word
        for syn in wn.synsets(word, pos = pos):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return random.choice(list(set(synonyms)))

    def get_replacable_word(self,seq : str,sep  : str = '[SEP]') -> str:
        seq = seq.replace(sep,'')
        seq = re.sub(r"[^a-zA-Z0-9]+", ' ', seq)
        seq = list(set(seq.split()))
        candidates = [cand for cand in seq if self.get_synonyms(cand) != []]

        return random.choice(candidates)
    
    def find_word_type(target_tag):
        word_type_dict = {wn.ADJ : ['JJ', 'JJR', 'JJS'],wn.ADV : ['RB', 'RBR', 'RBS']}
        for key, tag_list in word_type_dict.items():
            if target_tag in tag_list:
                return key
        return None  # Tag not found in any list

    def apply_syn_replacement(self,seq):
        word = self.get_replacable_word(seq)
        seq = seq.replace(word,random.choice(self.get_synonyms(word)))
        return seq

    def get_random_word(self, seq : str,sep :str = '[SEP]', min_len : int = 3, seed : int = 42) -> str:
        #random.seed(seed)
        seq = seq.replace(sep,'')
        seq = re.sub(r"[^a-zA-Z0-9]+", ' ', seq)
        seq = list(set(seq.split()))
        candidates = [word for word in seq if len(word) >= min_len]

        if(candidates == []):
            return None

        return random.choice(candidates)

    #changes one random letter of the word
    #dont interchange first or last letter
    def replace_letter(self, word : str, seed : int = 42)-> str:
        #random.seed(42)
        if len(word) <=2:
            return word
        idx = random.randint(1,len(word)-2)
        mod_word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx + 1:]
        return mod_word

    def apply_change_letter(self,seq : str, p : int = 0.3, sep :str = '[SEP]'):
        sentence = seq.split(sep)
        word = self.get_random_word(sentence[0],sep)
        if word == None or random.random() > p:
            return seq
        return sentence[0].replace(word,self.replace_letter(word)) + sep + sentence[1]



class STR_DataModule(pl.LightningDataModule):
    def __init__(self, train_data, batch_size=32, syn_replace = False, change_random_letter = False):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.syn_replace = syn_replace
        self.change_random_letter = change_random_letter

    def prepare_data_per_node(self):
    # Return True if you want prepare_data to be called on each node
        return False

    def prepare_data(self):
        train_df, val_df = train_test_split(self.train_data, test_size=0.2, random_state=42)
        self.train_dataset = str_dataset(train_df.reset_index(drop=True),syn_replace = self.syn_replace, change_random_letter = self.change_random_letter)
        self.val_dataset = str_dataset(val_df.reset_index(drop=True),syn_replace = False, change_random_letter = False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers = 2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers = 2)