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

nb_item=["100","250","500"]

d1 = torch.load("data/exp_paper_do.pt")
d2 = torch.load("data/exp_paper_doewc.pt")
#d3 = torch.load("data/exp_paper_doewc_ll.pt")

for nb in nb_item:
    plt.plot(d1[nb]["fitness"]["max"]["t=all"]["global"]["count1"],label="DO")
    plt.plot(d2[nb]["fitness"]["max"]["t=all"]["global"]["count1"],label="DO-EWC")
    #plt.plot(d3[nb]["fitness"]["max"]["t=all"]["global"]["count1"],label="DO-EWCR")
    plt.xlabel("Transitions")
    plt.ylabel("Succes Rate")
    plt.legend()
    plt.title("Comparison: "+nb)
    plt.xticks(torch.arange(0,21))
    plt.grid()
    plt.show()
    
    plt.plot(d1[nb]["fitness"]["max"]["t=all"]["global"]["mean"],label="DO")
    plt.plot(d2[nb]["fitness"]["max"]["t=all"]["global"]["mean"],label="DO-EWC")
    #plt.plot(d3[nb]["fitness"]["max"]["t=all"]["global"]["mean"],label="DO-EWCR")
    plt.xlabel("Transitions")
    plt.ylabel("Fitness Ratio")
    plt.legend()
    plt.title("Comparison: "+nb)
    plt.xticks(torch.arange(0,21))
    plt.grid()
    plt.show()

dim=["5","10","30"]
tight=["0.25","0.50","0.75"]

fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
        
table_do = [[dim[i]+"-"+tight[j],
             round((1-d1["100"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
             round((1-d2["100"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
             #round((1-d3["100"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,2),
             round((1-d1["250"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
             round((1-d2["250"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
             round((1-d1["500"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
             round((1-d2["500"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3)]
             #round((1-d3["250"]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,2)]
            for i in range(3) for j in range(3)]



table = plt.table(
    cellText=table_do,
    colLabels=["Problem Type",
               "DO: 100",
               "EWC: 100",
               #"EWCR: 100",
               "DO: 250",
               "EWC: 250",
               #"EWCR: 250",
               "DO: 500",
               "EWC: 500",], 
    loc='center', fontsize=12
)
"""
for i in range(3):
    for j in range(3):
        color = (j+2)/4 * (np.arange(3)==i)
        color += 0.3*(j+2)/4 * ((np.arange(3)-i)%3==2)

        color = np.append(color, 0.5)
        table[(1+3*i+j, 0)].set_facecolor(color)
        table[(1+3*i+j, 1)].set_facecolor(color)
        table[(1+3*i+j, 2)].set_facecolor(color)
        table[(1+3*i+j, 3)].set_facecolor(color)
        table[(1+3*i+j, 4)].set_facecolor(color)
        table[(1+3*i+j, 5)].set_facecolor(color)
        table[(1+3*i+j, 6)].set_facecolor(color)
"""
plt.title("Comparison of Fitness Ratio per Problem Type (\u2193)")
plt.figure(dpi=500)
plt.show()

import seaborn as sns

for nb in nb_item:
    sns.heatmap(d1[nb]["loss"]["global"]["mean"])
    plt.title("DO "+nb)
    plt.show()
    
    sns.heatmap(d2[nb]["loss"]["global"]["mean"])
    plt.title("DOEWC "+nb)
    plt.show()
    

