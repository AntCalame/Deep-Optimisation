import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import seaborn as sns
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)


L=["data/exp_furtherdim_ewc.pt",
   "data/exp_furtherdim_ewc_2.pt",
   "data/exp_furtherdim_classic.pt",
   "data/exp_furtherdim_classic_2.pt"]

for i in L:
    data = torch.load(i)
    for j in ["250","500"]:
        plt.hist(data[j]["time"]["t=-1"]["global"]["distribution"].reshape(-1)/60/60)
        
        plt.title(j +" - "+i)
        plt.show()
        print(max(data[j]["time"]["t=-1"]["global"]["distribution"].reshape(-1)/60/60))
        #print(max(data[j]["evaluation"]["t=-1"]["global"]["distribution"].reshape(-1)))
