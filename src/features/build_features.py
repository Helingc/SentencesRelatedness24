import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from gensim.models import FastText
import os

_FILE_ROOT = os.path.dirname(__file__)  # root of test folder
_SRC_ROOT = os.path.dirname(_FILE_ROOT)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project


df_str_rel = pd.read_csv(os.path.join(_PROJECT_ROOT,'data','raw','eng_train.csv'))
df_str_rel['Split_Text'] = df_str_rel['Text'].apply(lambda x: x.split("\n"))

fasttext_model = FastText.load_fasttext_format(os.path.join(_PROJECT_ROOT,'data','embeddings','cc.en.300.bin'))

def to_sent_emb(sentence):
    sentence_emb = np.array([fasttext_model.wv[word] for word in sentence.split() if word in fasttext_model.wv]).mean(axis=0)
    return sentence_emb



def cosine_similarity(vector_a, vector_b):
    vector_a = to_sent_emb(vector_a)
    vector_b = to_sent_emb(vector_b)
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    similarity = dot_product / (norm_a * norm_b)
    return similarity


df_str_rel['cos_sim'] = df_str_rel.apply(lambda row: cosine_similarity(row['Split_Text'][0],row['Split_Text'][1]), axis= 1)

true_scores = df_str_rel['Score'].values
pred_scores = df_str_rel['cos_sim'].values

print("Spearman Correlation:", round(spearmanr(true_scores,pred_scores)[0],2))