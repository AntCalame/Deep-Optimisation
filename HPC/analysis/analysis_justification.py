import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import seaborn as sns
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

data = torch.load("data/exp_justification_2.pt")


SSS=r""

class Analysis:
    def __init__(self,data):
        
        # 0-1 hill climbing
        # 0-2 upgraded
        D={}
        for i,j in data.items():
            s=""
            S=""
            if i=="0" or i=="1":
                s+="hc"
                S+="Hill Climbing algorithm"
                
            else:
                s+="ei"
                S+="Empty Initialization"
                
            s+="-"
            S+=" and "
            
            if i=="0" or i=="2":
                s+="ms"
                S+="Modified Search"
            
            else :
                s+="bs"
                S+="Basic Search"
                
            D[S]=j
            D[S]["label"]=s
            
        self.D=D
        self.dim = ["5","10","30"]
        self.tight = ["0.25","0.5","0.75"]
    
    def analysis(self):
        # Max
        fig, _ = plt.subplots(figsize=(10, 5))
        plt.rcParams.update({'font.size': 16})
        for (n,d) in self.D.items():
            plt.plot(torch.arange(1,21),
                     d["fitness"]["max"]["t=all"]["global"]["mean"],
                     label=d["label"])
        plt.legend(ncol=2,title="Possible Combinations")
        plt.xlabel("Transition")
        plt.xticks(torch.arange(1,21))
        plt.ylabel("Fitness Ratio")
        plt.title("Comparison of Fitness Ratio")
        plt.grid()
        plt.savefig(SSS+'\\fitness-ratio-all.pdf', format='pdf')
        plt.show()
        

        
        """
        # Time-evaluation
        for (n,d) in self.D.items():
            plt.plot(d["time"]["t=all"]["global"],d["evaluation"]["t=all"]["global"],
                     label=d["label"])
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Evaluations")
        plt.title("Evaluations per second")
        plt.grid()
        plt.show()
        
        for (n,d) in self.D.items():
            d_t=((d["time"]["t=all"]["global"]-d["time"]["t=all"]["global"].min())
                     /(d["time"]["t=all"]["global"].max()-d["time"]["t=all"]["global"].min()))
            d_e = ((d["evaluation"]["t=all"]["global"]-d["evaluation"]["t=all"]["global"].min())
             /(d["evaluation"]["t=all"]["global"].max()-d["evaluation"]["t=all"]["global"].min()))
            plt.plot(d_t,
                     d_e,
                     label=d["label"])
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Evaluations")
        plt.title("Evaluations per second")
        plt.grid()
        plt.show()
        
        for (n,d) in self.D.items():
            d_t=((d["time"]["t=all"]["global"]-d["time"]["t=all"]["global"].min())
                     /(d["time"]["t=all"]["global"].max()-d["time"]["t=all"]["global"].min()))
            
            d_e = ((d["evaluation"]["t=all"]["global"]-d["evaluation"]["t=all"]["global"].min())
             /(d["evaluation"]["t=all"]["global"].max()-d["evaluation"]["t=all"]["global"].min()))
            
            d_e-=d_t
            plt.plot(d_t,
                     d_e,
                     label=d["label"])
        plt.legend()
        plt.ylabel("Time (s)")
        plt.xlabel("Evaluations")
        plt.title("Time taken per evaluations")
        plt.grid()
        plt.show()
        
        for (n,d) in self.D.items():
            sns.heatmap(d["correlation"]["global"]["mean"][1:])
            plt.title(n)
            plt.show()
        """
        
A = Analysis(data)
A.analysis()
            