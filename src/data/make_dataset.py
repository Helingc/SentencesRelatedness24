import nltk
from nltk.corpus import wordnet
import random
import pandas as pd
import re
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import lightning as pl
from sklearn.model_selection import train_test_split




class str_dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, syn_replace = False, sep = '[SEP]'):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.syn_replace = syn_replace
        nltk.download('wordnet')

        self.dataframe['input'] = self.dataframe.apply(lambda row : row['Text'].replace('\n',sep),axis = 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the features and target from the DataFrame
        # Adjust this based on your DataFrame structure
        features = self.dataframe['input'].loc[idx]

        if self.syn_replace and random.random() < 0.3:
            features = self.apply_augmentation(features)

        token = self.tokenizer(features, return_tensors='pt', truncation='longest_first', padding='max_length',max_length=265)

        input_ids = token['input_ids'].squeeze()
        attention_mask = token['attention_mask'].squeeze()

        score = self.dataframe['Score'].loc[idx]

        score = torch.tensor(score, dtype=torch.float32).unsqueeze(dim=0)  # Adjust the dtype as needed

        return input_ids, attention_mask, score

    def get_synonyms(self,word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return list(set(synonyms))

    def get_replacable_word(self,seq,sep = '[SEP]'):
        seq = seq.replace(sep,'')
        seq = re.sub(r"[^a-zA-Z0-9]+", ' ', seq)
        seq = list(set(seq.split()))
        candidates = [cand for cand in seq if self.get_synonyms(cand) != []]

        return random.choice(candidates)

    def apply_augmentation(self,seq):
        word = self.get_replacable_word(seq)
        seq = seq.replace(word,random.choice(self.get_synonyms(word)))
        return seq



class STR_DataModule(pl.LightningDataModule):
    def __init__(self, train_data, batch_size=32, syn_replace = False):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = train_data
        self.syn_replace = syn_replace

    def prepare_data_per_node(self):
    # Return True if you want prepare_data to be called on each node
        return False

    def prepare_data(self):
        train_df, val_df = train_test_split(self.train_data, test_size=0.2, random_state=42)
        self.train_dataset = str_dataset(train_df.reset_index(drop=True),syn_replace = self.syn_replace)
        self.val_dataset = str_dataset(val_df.reset_index(drop=True),syn_replace = False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers = 2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers = 2)