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

train = torch.load("data/exp_tuning_replace_n1-n2.pt")
test = torch.load("data/exp_tuning_replace_n1-n2_test.pt")

SSS=r""

class Analysis:
    def __init__(self,train,test):
        self.train = train
        self.test = test
        
        self.dim = ["5","10","30"]
        self.tight = ["0.25","0.5","0.75"]
    
    def analysis(self):
        plt.rcParams.update({'font.size': 16})

        fisher="70000"
        
        j=self.train[fisher]
        l=self.test[fisher]
        
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(torch.arange(1,21),
                 j["fitness"]["max"]["t=all"]["global"]["mean"],
                 c='red')
        
        for i in range(20):
            ax.plot(torch.arange(1,21),
                     l["fitness"]["max"]["t=all"]["global"]["mean"][i],
                     c=[0,0,0.1+i*0.9/20])
        ax.set_xticks(torch.arange(1,21,2))
        ax.grid()
        
        blue_values = [(0, 0, 0.1 + 0.9 * i / 20) for i in range(20)]
        
        # Create a colorbar
        cmap = plt.cm.colors.ListedColormap(blue_values)
        norm = plt.cm.colors.BoundaryNorm(np.arange(-0.5, 20.5, 1), cmap.N)
        
        # Add colorbar to the plot
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
        cb.set_ticks(np.arange(0, 20, 2))
        cb.set_ticklabels(np.arange(0, 20, 2))
        cb.set_label("Transition")
        plt.ylabel("Fitness Ratio")
        plt.xlabel("Transition")
        plt.title("Optimisation Comparison - "+fisher)
        plt.legend()
        plt.savefig(SSS+'\\re-oc.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(torch.arange(1,21),
                 j["fitness"]["max"]["t=all"]["global"]["mean"],
                 c='red')
        
        for i in range(20):
            ax.scatter(i+1,
                    l["fitness"]["max"]["t=all"]["global"]["mean"][i][i],
                    c=[0,0,0.1+i*0.9/20])

        ax.set_xticks(torch.arange(1,21,2))
        ax.grid()
        
        blue_values = [(0, 0, 0.1 + 0.9 * i / 20) for i in range(20)]
        
        # Create a colorbar
        cmap = plt.cm.colors.ListedColormap(blue_values)
        norm = plt.cm.colors.BoundaryNorm(np.arange(-0.5, 20.5, 1), cmap.N)
        
        # Add colorbar to the plot
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
        cb.set_ticks(np.arange(0, 20, 2))
        cb.set_ticklabels(np.arange(0, 20, 2))
        cb.set_label("Transition")
        plt.title("Same Search - "+fisher)
        plt.ylabel("Fitness Ratio")
        plt.xlabel("Transition")
        plt.legend()
        plt.savefig(SSS+'\\re-ss.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for i in range(20):
            ax.scatter(i+1,
                    l["fitness"]["max"]["t=all"]["global"]["mean"][i][19],
                    c=[0,0,0.1+i*0.9/20])

        ax.set_xticks(torch.arange(1,21,2))
        ax.grid()
        
        blue_values = [(0, 0, 0.1 + 0.9 * i / 20) for i in range(20)]
        
        # Create a colorbar
        cmap = plt.cm.colors.ListedColormap(blue_values)
        norm = plt.cm.colors.BoundaryNorm(np.arange(-0.5, 20.5, 1), cmap.N)
        
        # Add colorbar to the plot
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
        cb.set_ticks(np.arange(0, 20, 2))
        cb.set_ticklabels(np.arange(0, 20, 2))
        cb.set_label("Transition")
        plt.ylabel("Fitness Ratio")
        plt.xlabel("Transition")
        plt.title("Maximum Search - "+fisher)
        plt.legend()
        plt.savefig(SSS+'\\re-ms.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        

A = Analysis(train,test)
A.analysis()
        
