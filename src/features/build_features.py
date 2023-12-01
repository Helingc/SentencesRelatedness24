import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from gensim.models import FastText
import os
from transformers import BertTokenizer, BertModel
import torch

import torch
import math

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

_FILE_ROOT = os.path.dirname(__file__)  # root of test folder
_SRC_ROOT = os.path.dirname(_FILE_ROOT)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project


df_str_rel = pd.read_csv(os.path.join(_PROJECT_ROOT,'data','raw','eng_train.csv'))
df_str_rel['Split_Text'] = df_str_rel['Text'].apply(lambda x: x.split("\n"))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def BERT_emb(sent, tokenizer = tokenizer, model = model):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Move the model to the GPU if available
    model.to(device)

    # Tokenize the input sentence
    tokens = tokenizer(sent, return_tensors='pt')

    # Move the input tokens to the GPU if available
    for key in tokens:
        tokens[key] = tokens[key].to(device)

    # Forward pass through the BERT model
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract embeddings from the last layer and calculate the mean
    last_hidden_states = torch.mean(outputs.last_hidden_state, dim=1).squeeze()

    return last_hidden_states

#fasttext_model = FastText.load_fasttext_format(os.path.join(_PROJECT_ROOT,'data','embeddings','cc.en.300.bin'))

# def to_sent_emb(sentence):
#     sentence_emb = np.array([fasttext_model.wv[word] for word in sentence.split() if word in fasttext_model.wv]).mean(axis=0)
#     return sentence_emb


# def cosine_similarity(vector_a, vector_b,sent_emb: callable):
#     vector_a = sent_emb(vector_a)
#     vector_b = sent_emb(vector_b)
#     dot_product = np.dot(vector_a, vector_b)
#     norm_a = np.linalg.norm(vector_a)
#     norm_b = np.linalg.norm(vector_b)

#     similarity = dot_product / (norm_a * norm_b)
#     return similarity


def cosine_similarity(vector_a, vector_b, sent_emb):
    # Assuming sent_emb is a PyTorch-compatible embedding function
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    vector_a = sent_emb(vector_a).to(device)
    vector_b = sent_emb(vector_b).to(device)
    
    dot_product = torch.dot(vector_a, vector_b)
    norm_a = torch.norm(vector_a)
    norm_b = torch.norm(vector_b)
    
    similarity = dot_product / (norm_a * norm_b)
    
    print()

    return similarity.item()



df_str_rel['cos_sim'] = df_str_rel.apply(lambda row: cosine_similarity(row['Split_Text'][0],row['Split_Text'][1],BERT_emb), axis= 1)

true_scores = df_str_rel['Score'].values
pred_scores = df_str_rel['cos_sim'].values

print("Spearman Correlation:", round(spearmanr(true_scores,pred_scores)[0],2))