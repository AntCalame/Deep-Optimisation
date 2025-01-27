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

#s, spe = "replace", "70000"
#s, spe = "all", "70000"
#s, spe = "classic", "0"

L=[["replace", "70000"],
   ["all", "70000"],
   ["classic", "0"]]

for X in L:
    s, spe = X[0], X[1]

    train = torch.load("data/exp_retrain_"+s+".pt")[spe]
    test = torch.load("data/exp_retrain_"+s+"_test.pt")[spe]
    
    plt.plot(torch.arange(1,21),
             train['fitness']['max']['t=all']['global']['mean'],
             label="train",c="red")
    for j in range(20):
        plt.scatter(j+1,
                 test['fitness']['max']['t=all']['global']['mean'][j][-1],
                 c = "blue")
    plt.xticks(torch.arange(1,21))
    plt.legend()
    plt.grid()
    plt.title("Comparison Train-Retrain: "+s)
    plt.xlabel("Transition")
    plt.ylabel("Fitness Ratio")
    plt.show()
    
    for j in range(20):
        plt.scatter(j+1,
                 test['fitness']['max']['t=all']['global']['mean'][j][-1],
                 c = "blue")
    plt.xticks(torch.arange(1,21))
    plt.legend()
    plt.grid()
    plt.title("Fitness Ratio Retrain: "+s)
    plt.xlabel("Transition")
    plt.ylabel("Fitness Ratio")
    plt.show()
    
    for j in range(20):
        plt.scatter(j+1,
                 test['fitness']['max']['t=all']['global']['count1'][j][-1],
                 c = "blue")
    plt.xticks(torch.arange(1,21))
    plt.legend()
    plt.grid()
    plt.title("Success Rate Retrain: "+s)
    plt.xlabel("Transition")
    plt.ylabel("Success Rate")
    plt.show()
    
    """
    plt.scatter(train['time']['t=all']['global'],
             train['fitness']['max']['t=all']['global']['mean'],
             label="train",c="red")
    
    for j in range(20):
        plt.scatter(train['time']['t=all']['global'][-1]
                    +test['time']['t=-1']['global']["mean"][j],
                    test['fitness']['max']['t=all']['global']['mean'][j][-1], 
                    c = "blue")
        
    plt.plot(train['time']['t=all']['global'][-1]
                +test['time']['t=-1']['global']["mean"],
                test['fitness']['max']['t=all']['global']['mean'][:,-1], 
                c = "blue")
    """
