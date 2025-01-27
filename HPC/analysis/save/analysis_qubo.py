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

L1 = ["", "_500", "_1000"]
L2 = ["100", "500", "1000"]
L3 = [20,10,10]

for (s,ss,k) in zip(L1,L2,L3):
    d1 = torch.load("data/exp_qubo_classic"+s+".pt")["0"]
    d2 = torch.load("data/exp_qubo_replace"+s+".pt")["70000"]
    d3 = torch.load("data/exp_qubo_replacem"+s+".pt")["70000"]
    
    
    plt.plot(d1["fitness"]["max"]["t=all"]["global"]["mean"],label="classic")
    plt.plot(d2["fitness"]["max"]["t=all"]["global"]["mean"],label="ewc")
    plt.plot(d3["fitness"]["max"]["t=all"]["global"]["mean"],label="ewc + L1L2Reg")
    plt.legend()
    plt.xlabel("Transitions")
    plt.ylabel("Fitness Ratio")
    plt.xticks(torch.arange(0,k))
    plt.grid()
    plt.title("Qubo: "+ss+": Fitness ratio")
    plt.show()
    
    plt.plot(d1["fitness"]["max"]["t=all"]["global"]["count1"],label="classic")
    plt.plot(d2["fitness"]["max"]["t=all"]["global"]["count1"],label="ewc")
    plt.plot(d3["fitness"]["max"]["t=all"]["global"]["count1"],label="ewc + L1L2Reg")
    plt.legend()
    plt.xlabel("Transitions")
    plt.ylabel("Succes Rate")
    plt.xticks(torch.arange(0,k))
    plt.grid()
    plt.title("Qubo: "+ss+": Succes Rate")
    plt.show()
    
    plt.plot(d1["transition_loss"]["recon"]["global"][0],label="classic",alpha=0.5)
    plt.plot(d2["transition_loss"]["recon"]["global"][0],label="ewc",alpha=0.5)
    plt.plot(d3["transition_loss"]["recon"]["global"][0],label="ewc + L1L2Reg",alpha=0.5)
    plt.legend()
    plt.title("Loss at 1 transition")
    plt.show()
    
    sns.heatmap(d1["loss"]["global"]["mean"])
    plt.title("Qubo ("+ss+") DO")
    plt.show()
    
    sns.heatmap(d2["loss"]["global"]["mean"])
    plt.title("Qubo ("+ss+") EWC")
    plt.show()
    
    sns.heatmap(d3["loss"]["global"]["mean"])
    plt.title("Qubo ("+ss+") EWC + L1L2")
    plt.show()
    
    for d in [d1,d2,d3]:
        L_loss=[]
        L_label=[]
                
        for i,j in d["transition_loss"].items():
            if i!="loss":
                if i=="l2":
                    for k in range(len(L_loss)):
                        L_loss[k]+=j['global'].reshape(-1)*0.0001
                        
                    L_loss+=[j['global'].reshape(-1)*0.0001]
                    L_label+=[i]
                else:
                    for k in range(len(L_loss)):
                        L_loss[k]+=j['global'].reshape(-1)
                        
                    L_loss+=[j['global'].reshape(-1)]
                    L_label+=[i]
        fig, _ = plt.subplots(figsize=(10, 5))
        for i in range(len(L_loss)):
            plt.fill_between(torch.arange(0,j['global'].size(0)*j['global'].size(1))/j['global'].size(1),
                             0,
                             L_loss[i],
                             label=L_label[i])
            
        plt.xticks(torch.arange(0,21))
        plt.xlabel("Succes Rate")
        plt.ylabel("Loss")
        plt.legend(title="Component: ")
        plt.title("Detailed loss per component: ")
        plt.grid()
        plt.show()

    