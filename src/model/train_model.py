import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


import pandas as pd
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data.make_dataset import STR_DataModule,str_dataset
from src.model.model import SentenceSimilarityModel
from sklearn.model_selection import KFold

    


if __name__ == '__main__':

    kf = KFold(n_splits=5, shuffle=True, random_state=42)




    train_data = pd.read_csv('/content/eng_train.csv')

    all_spearman_corrs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):

      str_datamodule = STR_DataModule(train_data = train_data.iloc[train_idx], batch_size = 32, syn_replace = True)

      model = SentenceSimilarityModel()

      trainer = pl.Trainer(max_epochs=30,precision="16-mixed",callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],accelerator='gpu')

      # tuner = pl.tuner.Tuner(trainer)
      # lr_finder = tuner.lr_find(model, datamodule = str_datamodule, min_lr = 1e-7,max_lr= 1e-2)

      # # Pick point based on plot, or get suggestion
      # new_lr = lr_finder.suggestion()

      # #  update hparams of the model
      # model.lr = new_lr

      trainer.fit(model, datamodule=str_datamodule)

      val_set = str_dataset(train_data.iloc[val_idx])

      val_dataloader = DataLoader(val_set, batch_size = 16, num_workers = 2)


      trainer.test(model, val_dataloader)


      # Calculate and print the average Spearman correlation
      average_spearman_corr = trainer.callback_metrics["val_spearman"].mean()
      print(
          f"Average Spearman Correlation for Fold {fold + 1}: {average_spearman_corr}"
      )

      all_spearman_corrs.append(average_spearman_corr)


    # Calculate and print the overall average Spearman correlation
    overall_average_spearman_corr = sum(all_spearman_corrs) / len(all_spearman_corrs)
    print(
        f"Overall Average Spearman Correlation across all folds: {overall_average_spearman_corr}"
    )
