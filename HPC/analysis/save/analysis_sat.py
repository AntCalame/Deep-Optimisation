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

d1 = torch.load("data/exp_sat_classic.pt")["0"]
d3 = torch.load("data/exp_sat_replacem.pt")["70000"]
d2 = torch.load("data/exp_sat_replace.pt")["70000"]
d4 = torch.load("data/exp_sat_replace_2.pt")["70000"]
d5 = torch.load("data/exp_sat_replace_3.pt")["70000"]

plt.plot(d1["fitness"]["max"]["t=all"]["global"]["mean"],label="classic")
plt.plot(d2["fitness"]["max"]["t=all"]["global"]["mean"],label="ewc")
plt.plot(d3["fitness"]["max"]["t=all"]["global"]["mean"],label="ewc + L1L2Reg")
plt.plot(d4["fitness"]["max"]["t=all"]["global"]["mean"],label="ewc2")
plt.plot(d5["fitness"]["max"]["t=all"]["global"]["mean"],label="ewc3")
plt.legend()
plt.xlabel("Transitions")
plt.ylabel("Fitness Ratio")
plt.xticks(torch.arange(0,21))
plt.grid()
plt.title("Sat: Fitness ratio")
plt.show()

plt.plot(d1["fitness"]["max"]["t=all"]["global"]["count1"],label="classic")
plt.plot(d2["fitness"]["max"]["t=all"]["global"]["count1"],label="ewc")
plt.plot(d3["fitness"]["max"]["t=all"]["global"]["count1"],label="ewc + L1L2Reg")
plt.plot(d4["fitness"]["max"]["t=all"]["global"]["count1"],label="ewc2")
plt.plot(d5["fitness"]["max"]["t=all"]["global"]["count1"],label="ewc3")
plt.legend()
plt.xlabel("Transitions")
plt.ylabel("Succes Rate")
plt.xticks(torch.arange(0,21))
plt.grid()
plt.title("Sat: Fitness ratios")
plt.show()

sns.heatmap(d1["loss"]["global"]["mean"])
plt.title("Sat: DO")
plt.show()

sns.heatmap(d2["loss"]["global"]["mean"])
plt.title("Sat: EWC")
plt.show()

sns.heatmap(d3["loss"]["global"]["mean"])
plt.title("Sat: EWC + L1L2")
plt.show()

sns.heatmap(d4["loss"]["global"]["mean"])
plt.title("Sat: EWC2")
plt.show()

sns.heatmap(d5["loss"]["global"]["mean"])
plt.title("Sat: EWC3")
plt.show()

for d in [d1,d2,d3,d4,d5]:
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