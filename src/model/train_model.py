import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from transformers import DebertaTokenizer, DebertaForSequenceClassification,AutoTokenizer

import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class SentenceSimilarityModel(pl.LightningModule):
    def __init__(self, learning_rate=2e-5):
        super(SentenceSimilarityModel, self).__init__()
        self.deberta = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-v3-small',ignore_mismatched_sizes=True,num_labels=1,output_hidden_states= True)

        self.custom_layer = torch.nn.Linear(in_features=self.deberta.config.hidden_size, out_features=1)
        #Freeze parameters of the DeBERTa model
        for param in self.deberta.parameters():
            param.requires_grad = False
        self.learning_rate = learning_rate

    def forward(self, input_ids,attention_mask):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)

        hidden_state = outputs.hidden_states[-1]

        logits = self.custom_layer(hidden_state[:, 0, :])
        return torch.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, score = batch
        logits = self(input_ids,attention_mask)

        
        loss = F.mse_loss(logits, score)
        spearman = spearmanr(logits.detach().cpu().numpy(),score.detach().cpu().numpy()).statistic

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("spearman", spearman, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, score = batch
        logits = self(input_ids, attention_mask)

        
        loss = F.mse_loss(logits, score)

        spearman = spearmanr(logits.detach().cpu().numpy(),score.detach().cpu().numpy()).statistic

        self.log("val_loss", loss,on_epoch = True, prog_bar = True)
        self.log("spearman",spearman,on_epoch = True, prog_bar = True)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.custom_layer.parameters(), lr=self.learning_rate)



class str_dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the features and target from the DataFrame
        # Adjust this based on your DataFrame structure
        features = self.dataframe['input'].loc[idx]
        token = self.tokenizer(features, return_tensors='pt', truncation='longest_first', padding='max_length',max_length=265)

        input_ids = token['input_ids'].squeeze()
        attention_mask = token['attention_mask'].squeeze()

        score = self.dataframe['Score'].loc[idx]

        score = torch.tensor(score, dtype=torch.float32).unsqueeze(dim=0)  # Adjust the dtype as needed

        return input_ids, attention_mask, score
    


class STR_DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=16):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,num_workers = 2,persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,num_workers = 2,persistent_workers=True)



if __name__ == '__main__':
    train_data = pd.read_csv('/Users/lemarx/Documents/01_projects/SentencesRelatedness24/data/raw/eng_train.csv')
    sep = '[SEP]'
    test = train_data['Text'].loc[0].replace('\n',sep)
    train_data['input'] = train_data.apply(lambda row : row['Text'].replace('\n',sep),axis = 1)

    train_df, val_df = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = str_dataset(train_df.reset_index(drop=True))
    val_dataset = str_dataset(val_df.reset_index(drop=True))

    str_datamodule = STR_DataModule(train_dataset=train_dataset, val_dataset= val_dataset, batch_size = 16)


    model = SentenceSimilarityModel()

    trainer = pl.Trainer(max_epochs=25,precision="16-mixed",callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],accelerator='gpu')

    trainer.fit(model, datamodule=str_datamodule)