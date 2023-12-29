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
    def __init__(self, dataframe, syn_replace = False, change_random_letter = False, sep = '[SEP]',seed = 42):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.syn_replace = syn_replace
        self.change_random_letter = change_random_letter
        self.seed = seed

        #donwload nltk data
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')

        #replace the dataset seperator with the usual custom Seperator Token
        self.dataframe['input'] = self.dataframe.apply(lambda row : row['Text'].replace('\n',sep),axis = 1)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the features and target from the DataFrame
        # Adjust this based on your DataFrame structure
        features = self.dataframe['input'].loc[idx]

        if self.syn_replace:
            features = self.apply_syn_replacement(features)
        if self.change_random_letter:
            features = self.apply_change_letter(features)

        token = self.tokenizer(features, return_tensors='pt', truncation='longest_first', padding='max_length',max_length=265)

        input_ids = token['input_ids'].squeeze()
        attention_mask = token['attention_mask'].squeeze()

        score = self.dataframe['Score'].loc[idx]

        score = torch.tensor(score, dtype=torch.float32).unsqueeze(dim=0)  # Adjust the dtype as needed

        return input_ids, attention_mask, score
    
    #takes the first sentence of the 2 sentences which are compared as an argument
    #returns a tuple of of the chosen word and pos tag
    def get_random_word(self,seq : str, min_len : int = 3,seed : int = 42, sep = '[SEP]', syn_replace : bool = False) -> tuple:
        random.seed(seed)
        seq = seq.replace(sep, '')
        tokens = nltk.word_tokenize(seq)
        pos_tags = nltk.pos_tag(tokens)
        if syn_replace:
            pos_tags = self.syn_replace_choice(pos_tags)
        
        pos_tags = list(set(pos_tags))
        candidates = [word for word in pos_tags if len(word[0]) >= min_len]
        if(candidates == []):
            return None

        return random.choice(candidates)
    
    #takes a pos_tagged sentence as function argument
    def syn_replace_choice(self,pos_tags : list):
        replacable_tags = ['JJ', 'JJR', 'JJS','RB', 'RBR', 'RBS']
        filtered_data = [item for item in pos_tags if item[1] in replacable_tags]
        return filtered_data

    def get_synonym(self,word : str, pos : str, seed : int = 42) -> str:
        synonyms = []
        if pos == None:
            return word
        for syn in wn.synsets(word, pos = pos):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        random.seed(seed)
        return random.choice(list(set(synonyms)))
    
    def find_word_type(self,target_tag):
        word_type_dict = {wn.ADJ : ['JJ', 'JJR', 'JJS'],wn.ADV : ['RB', 'RBR', 'RBS']}
        for key, tag_list in word_type_dict.items():
            if target_tag in tag_list:
                return key
        return None  # Tag not found in any list

    def apply_syn_replacement(self,seq : str, sep : str = '[SEP]', seed : int = 42, p : float = 0.3):
        random.seed(seed)
        if random.random() > p:
            return seq
        word = self.get_random_word(seq, sep = sep, syn_replace= True, seed = seed)
        seq = seq.replace(word[0],self.get_synonym(word[0], pos = self.find_word_type(word[1]), seed = seed))
        return seq


    #changes one random letter of the word
    #dont interchange first or last letter
    def replace_letter(self,word : str, seed : int = 42)-> str:
        random.seed(seed)
        if len(word) <=2:
            return word
        idx = random.randint(1,len(word)-2)
        mod_word = word[:idx] + random.choice(string.ascii_lowercase) + word[idx + 1:] 
        return mod_word

    def apply_change_letter(self,seq : str, p : int = 0.3, sep :str = '[SEP]', seed = 42):
        seq = seq.split(sep)
        word = self.get_random_word(seq[0])[0]
        random.seed(seed)
        if word == None or random.random() < p:
            return seq[0] + sep + seq[1]
        return seq[0].replace(word,self.replace_letter(word,seed = seed)) + sep + seq[1]


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