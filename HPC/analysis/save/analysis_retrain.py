import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import seaborn as sns
import os
import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

L=[
   ["all","70000"],
   ["replace","70000"],
   ["classic","0"]
   ]

for X in L:
    s=X[0]
    spe=X[1]

    train = torch.load("data/exp_retrain_"+s+".pt")[spe]
    test = torch.load("data/exp_retrain_"+s+"_test.pt")[spe]
    
    plt.plot(torch.arange(1,21),
             train['fitness']['max']['t=all']['global']['mean'],
             label="train",c="red")
    for i in range(5):
        j = 4*i +1
        plt.plot(torch.arange(1,21),
                 test['fitness']['max']['t=all']['global']['mean'][i],
                 label=str(j))
    plt.xticks(torch.arange(1,21))
    plt.legend()
    plt.grid()
    plt.title(s)
    plt.xlabel("Transition")
    plt.ylabel("Fitness Ratio")
    plt.show()
    
    plt.plot(torch.arange(10,21),
             train['fitness']['max']['t=all']['global']['mean'][9:],
             label="train",c="red")
    for i in range(5):
        j = 4*i +1
        plt.plot(torch.arange(10,21),
                 test['fitness']['max']['t=all']['global']['mean'][i][9:],
                 label=str(j))
    plt.title(s+" (Zoomed in)")
    plt.xticks(torch.arange(10,21))
    plt.legend()
    plt.ylabel("Fitness Ratio")
    plt.grid()
    plt.xlabel("Transition")
    plt.show()

    plt.plot(torch.arange(1,21),
             train['fitness']['max']['t=all']['global']['count1'],
             label="train",c="red")
    for i in range(5):
        j = 4*i +1
        plt.plot(torch.arange(1,21),
                 test['fitness']['max']['t=all']['global']['count1'][i],
                 label=str(j))
    plt.xticks(torch.arange(1,21))
    plt.legend()
    plt.grid()
    plt.title(s)
    plt.xlabel("Transition")
    plt.ylabel("Succes Rate")
    plt.show()
    
    plt.plot(torch.arange(10,21),
             train['fitness']['max']['t=all']['global']['count1'][9:],
             label="train",c="red")
    for i in range(5):
        j = 4*i +1
        plt.plot(torch.arange(10,21),
                 test['fitness']['max']['t=all']['global']['count1'][i][9:],
                 label=str(j))
    plt.title(s+" (Zoomed in)")
    plt.xticks(torch.arange(10,21))
    plt.legend()
    plt.ylabel("Succes Rate")
    plt.grid()
    plt.xlabel("Transition")
    plt.show()