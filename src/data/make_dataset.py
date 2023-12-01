import pandas as pd
import os
import numpy as np


_FILE_ROOT = os.path.dirname(__file__)  # root of test folder
_SRC_ROOT = os.path.dirname(_FILE_ROOT)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project


df_str_rel = pd.read_csv(os.path.join(_PROJECT_ROOT,'data','raw','eng_train.csv'))
df_str_rel['Split_Text'] = df_str_rel['Text'].apply(lambda x: x.split("\n"))

df_str_rel[['Split_Text','Score']].to_csv(os.path.join(_PROJECT_ROOT,'data','processed','train.csv'))




