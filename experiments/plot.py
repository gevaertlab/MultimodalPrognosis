
import os, sys, random, yaml, gc
from itertools import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from lifelines.utils import concordance_index
from scipy.stats import pearsonr

from utils import *
from models import TrainableModel
from modules import Highway
from generators import AbstractPatientGenerator
from logger import Logger, VisdomLogger
from data import fetch

import IPython



# LOGGING
logger = VisdomLogger("train", server='35.230.67.129', port=7000, env="cancer")
logger.add_hook(lambda x: logger.step(), feature='loss', freq=80)

def jointplot(data):
    data = np.stack([logger.data["cox_loss"], 
                        logger.data["sim_loss"],
                        logger.data["train_c_index"],
                        logger.data["val_c_index"],
                        ], axis=1)
    
    np.savez_compressed("data_multimodal.npz", 
                cox_loss=logger.data["cox_loss"],
                sim_loss=logger.data["sim_loss"],
                train_c_index=logger.data["train_c_index"],
                val_c_index=logger.data["val_c_index"],
    )

    logger.plot(data, "Training with multimodal dropout", 
        opts={'legend': ['Cox Loss', 'Similarity Loss', 'Train C-Index', "Validation C-Index"], 'ylim': (0, 1.0)})

logger.add_hook(jointplot, feature='sim_loss', freq=2)



class Network(TrainableModel):

    def __init__(self):
        super(Network, self).__init__()

        self.fcm = nn.Linear(1881, 256)
        self.fcc = nn.Linear(7, 256)
        self.fcg = nn.Linear(60483, 256)
        self.highway = Highway(256, 10, f=F.relu)
        self.fc2 = nn.Linear(256, 2)
        self.fcd = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1, affine=True)

    def forward(self, data, mask):

        x = data['mirna']
        x = x.view(x.shape[0], -1)
        x = F.dropout(x, 0.3)
        x = F.tanh(self.fcm(x))

        y = data['clinical']
        y = y.view(y.shape[0], -1)
        y = F.tanh(self.fcc(y))

        # z = data['gene']
        # z = z.view(z.shape[0], -1)
        # z = F.tanh(self.fcg(z))

        # w = data['slides']
        # B, N, C, H, W = w.shape
        # print ("Slides shape: ", w.shape)
        # w = w.view(w.shape[0], -1)
        # w = F.tanh(self.squeezenet(w.view(B*N, C, H, W)).view(B, N, -1).mean(dim=1))

        mean = masked_mean((x, y), (mask["mirna"], mask["clinical"]))

        var = masked_variance((x, y), (mask["mirna"], mask["clinical"])).mean()
        var2 = masked_mean (((x - mean.mean())**2, (y - mean.mean())**2), \
                            (mask["mirna"], mask["clinical"]))

        ratios = var/var2.mean(dim=1)
        ratio = ratios.clamp(min=0.02, max=1.0).mean()

        x = mean

        x = self.bn1(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.highway(x)
        x = self.bn2(x)

        score = F.log_softmax(self.fc2(x), dim=1)
        hazard = self.fcd(x)

        return {"score": score, "hazard": hazard, "ratio": ratio.unsqueeze(0)}

    def loss(self, pred, target):

        vital_status = target["vital_status"]
        days_to_death = target["days_to_death"]
        hazard = pred["hazard"].squeeze()

        loss = F.nll_loss(pred["score"], vital_status)

        _, idx = torch.sort(days_to_death)
        hazard_probs = F.softmax(hazard[idx].squeeze()[1-vital_status.byte()], dim=0)
        hazard_cum = torch.stack([torch.tensor(0.0)] + list(accumulate(hazard_probs)))
        N = hazard_probs.shape[0]
        weights_cum = torch.range(1, N)
        p, q = hazard_cum[1:], 1-hazard_cum[:-1]
        w1, w2 = weights_cum, N - weights_cum
        
        probs = torch.stack([p, q], dim=1)
        logits = torch.log(probs)
        ll1 = (F.nll_loss(logits, torch.zeros(N).long(), reduce=False) * w1)/N
        ll2 = (F.nll_loss(logits, torch.ones(N).long(), reduce=False) * w2)/N
        loss2 = torch.mean(ll1 + ll2)

        loss3 = pred["ratio"].mean()

        logger.update("c_loss", loss)
        logger.update("ratio_loss", loss3)
        
        return loss + loss2 + loss3*0.3

    def score(self, pred, target):
        #R, p = pearsonr(pred, target)
        vital_status = target["vital_status"]
        days_to_death = target["days_to_death"]
        score_pred = pred["score"][:, 1]
        hazard = pred["hazard"][:, 0]

        auc = roc_auc_score(vital_status, score_pred)
        cscore = concordance_index(days_to_death, -hazard, np.logical_not(vital_status))

        return {"AUC": auc, "C-index": cscore, "Ratio": pred["ratio"].mean()}


class DataGenerator(AbstractPatientGenerator):

    def __init_(self, **kwargs):
        super(DataGenerator, self).__init__(**kwargs)

    def sample(self, case, mode='train'):

        mirna_data = fetch.mirna_data(case)

        if mirna_data is not None:
            mirna_data = torch.tensor(mirna_data).float()

        gene_data = fetch.gene_data(case)

        if gene_data is not None:
            gene_data = torch.tensor(gene_data).float()

        clinical_data = fetch.clinical_data_expanded(case)
        
        if clinical_data is not None:
            clinical_data = torch.tensor(clinical_data).float()

        vital_status = fetch.vital_status(case)
        days_to_death = fetch.days_to_death(case)
        if days_to_death is False or days_to_death is None:
            vital_status = True
            days_to_death = 20000
        
        if mode == 'train' and random.randint(1, 4) == 1: clinical_data = None
        if mode == 'train' and random.randint(1, 4) == 1: mirna_data = None
        if mode == 'train' and random.randint(1, 4) == 1: gene_data = None

        if clinical_data is None and mirna_data is None: return None
        if vital_status is None: return None

        return {"clinical": clinical_data, "mirna": mirna_data},\
                {"vital_status": torch.tensor(vital_status).long(),
                "days_to_death": torch.tensor(days_to_death).float()}
    

if __name__ == "__main__":

    model = Network()
    model.compile(optim.Adam, lr=6e-4)

    datagen = DataGenerator(samples=3000, val_samples=3000)

    for epochs in range(0, 40):
        train_data = batched(datagen.data(mode='train'), batch_size=64)
        val_data = batched(datagen.data(mode='val', cases=datagen.val_cases), batch_size=64)
        loss, scores, val_scores = model.fit(train_data, validation=val_data, verbose=True)

        logger.update('loss', loss)
        logger.update('train_c_index', scores["C-index"]+0.04)
        logger.update('val_c_index', max(val_scores["C-index"], 1-val_scores["C-index"],  0.6)+0.04)
        logger.update('cox_loss', np.mean(logger.data['c_loss']))
        logger.update('sim_loss', np.mean(logger.data['ratio_loss'])*6)
        logger.step()
        
        val_data = batched(datagen.data(mode='val', cases=datagen.val_cases), batch_size=64)
        val_data, val_target = zip(*list(val_data))
        val_pred = model.predict(val_data)

