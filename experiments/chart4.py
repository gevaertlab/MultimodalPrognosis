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
from data import fetch

import IPython

#For miRNA, clinical and slide data

MULTIMODAL = sys.argv[1]
OUTFILE = f"results/{MULTIMODAL}_mirna_clinical_slide.txt"
print (OUTFILE)

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
        x = F.dropout(x, 0.4)
        x = F.tanh(self.fcm(x))

        y = data['clinical']
        y = y.view(y.shape[0], -1)
        y = F.tanh(self.fcc(y))

        w = data['slides']
        B, N, C, H, W = w.shape
        print ("Slides shape: ", w.shape)
        w = w.view(w.shape[0], -1)
        w = F.tanh(self.squeezenet(w.view(B*N, C, H, W)).view(B, N, -1).mean(dim=1))


        mean = masked_mean((x, y, w), (mask["mirna"], mask['clinical'], mask["gene"]))

        var = masked_variance((x, y, w), (mask["mirna"], mask['clinical'], mask["gene"])).mean()
        var2 = masked_mean (((x - mean.mean())**2, (y - mean.mean())**2, (w - mean.mean())**2), \
                            (mask["mirna"], mask["clinical"], mask["gene"]))

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
        hazard_probs = F.softmax(hazard[idx].squeeze()[1-vital_status.byte()])
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
        
        return loss + loss2 + loss3*0.3

    def score(self, pred, target):
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
        
        if MULTIMODAL == 'multi':
            if mode == 'train' and random.randint(1, 4) == 1: clinical_data = None
            if mode == 'train' and random.randint(1, 4) == 1: mirna_data = None
            if mode == 'train' and random.randint(1, 4) == 1: gene_data = None

        if clinical_data is None and mirna_data is None and gene_data is None: return None
        if vital_status is None: return None

        return {"clinical": clinical_data, "gene": gene_data, "mirna": mirna_data},\
                {"vital_status": torch.tensor(vital_status).long(),
                "days_to_death": torch.tensor(days_to_death).float()}
    

if __name__ == "__main__":

    print ("HELLO")

    model = Network()
    model.compile(optim.Adam, lr=8e-4)

    datagen = DataGenerator(samples=40000, val_samples=10000)

    for epochs in range(0, 5):
        train_data = batched(datagen.data(mode='train'), batch_size=64)
        val_data = batched(datagen.data(mode='val', cases=datagen.val_cases), batch_size=64)
        model.fit(train_data, validation=val_data, verbose=True)

        stratified = {}
        for case in datagen.train_cases:
            cancer_type = fetch.cancer_type(case)
            cancer_type = fetch.disease_lookup[cancer_type]
            if cancer_type is None: continue
            stratified.setdefault(cancer_type, []).append(case)

        with open(OUTFILE, "w") as outfile:
            for cancer_type, cases in sorted(stratified.items()):
                print (cancer_type, len(cases))
                test_data = batched(datagen.data(mode='val', cases=cases), batch_size=64)

                try:
                    score = model.eval_data(test_data)
                except:
                    score = 0.0
                print (f"{cancer_type}, {score}", file=outfile)

    #model.save("results/predict11.pth") 
