from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from scipy.stats import spearmanr, pearsonr
import lightning as pl
import torch.functional as F

from src.data.data_augmentation import apply_interpol

class SentenceSimilarityModel(pl.LightningModule):
    def __init__(self, learning_rate=1.737e-6):
        super(SentenceSimilarityModel, self).__init__()
        self.berta = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',ignore_mismatched_sizes=True,num_labels=1)


        self.learning_rate = learning_rate

    def forward(self, input_ids,attention_mask):
        outputs = self.berta(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        return torch.sigmoid(logits)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, score = batch
        logits = self(input_ids,attention_mask)


        loss = F.mse_loss(logits, score)
        spearman = spearmanr(logits.detach().cpu().numpy(),score.detach().cpu().numpy()).statistic

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_spearman", spearman,on_step = False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, score = batch
        logits = self(input_ids, attention_mask)


        loss = F.mse_loss(logits, score)

        spearman = spearmanr(logits.detach().cpu().numpy(),score.detach().cpu().numpy()).statistic

        self.log("val_loss", loss,on_epoch = True, prog_bar = True)
        self.log("val_spearman",spearman,on_epoch = True, prog_bar = True)


    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, score = batch
        logits = self(input_ids, attention_mask)

        # print("Score:", score)
        # print('Score_var:', score.var())

        loss = F.mse_loss(logits, score)
        spearman = spearmanr(logits.detach().cpu().numpy(), score.detach().cpu().numpy()).statistic

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_spearman", spearman, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, fused = True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-8)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}

