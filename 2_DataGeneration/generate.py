import numpy as np
import pandas as pd
import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
import sys
import torch

# Charger les données
print("Chargement des données...")
df = pd.read_csv('Big_Data_Strokdem_imputed.csv')

log.add(sink=sys.stderr, level="INFO")

#Preprocessing data with onehot
print('Preprocessing data with onehot...')
loader = GenericDataLoader(
    df
)

Plugins().list()

#Load a synthetic model
print('Load a synthetic model...')
syn_model = Plugins().get("marginal_distributions")
syn_model.fit(loader)

#Generate synthetic data
print('Generate synthetic data...')
syn_model = Plugins().get("adsgan")
syn_model.fit(loader)

count = 10
syn_model.generate(
    count=count, cond=np.ones(count)
).dataframe()