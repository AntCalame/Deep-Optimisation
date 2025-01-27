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

d = torch.load("data/ex_paper_gomea.pt")

results = {}
results2 = {}

for i,j in d.items():
    if not i[4:] in results.keys():
        results[i[4:]] = {}
        results2[i[4:]] = {}
    results[i[4:]][i[:3]] = round((1-j["fitness"]["max"]["global"]["mean"].item())*100,3)
    results2[i[4:]][i[:3]] =j["fitness"]["max"]["macro"]["mean"]
    
d_do = torch.load("data/exp_paper_do.pt")
d_doe = torch.load("data/exp_paper_doewc.pt")

map_name = {"0" : "Univariate",
            "2" : "MI-NF",
            "3" : "MI-F",
            "4" : "NMI-NF",
            "5" : "NMI-F"}

l_item=["100", "250", "500"]

table = [
    [results[i][j]
     for j in l_item
     ] for i in map_name.keys()
           ]

l_do = [round((1-d_do[i]["fitness"]["max"]["t=-1"]["macro"]["mean"][:,1:].mean().item())*100,3) 
        for i in l_item]
l_doe = [round((1-d_doe[i]["fitness"]["max"]["t=-1"]["macro"]["mean"][:,1:].mean().item())*100,3) 
        for i in l_item]

l_do = [round((1-d_do[i]["fitness"]["max"]["t=-1"]["macro"]["mean"].mean().item())*100,3) 
        for i in l_item]
l_doe = [round((1-d_doe[i]["fitness"]["max"]["t=-1"]["macro"]["mean"].mean().item())*100,3) 
        for i in l_item]

table+=[l_do]
table+=[l_doe]

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')
table = plt.table(
    cellText=table,
    colLabels=["100 items",
               "250 items",
               "500 items",],
    rowLabels=["Gomea "+map_name[str(k)] for k in [0,2,3,4,5]]+["DO","DO-EWC"],
    loc='center',
    cellLoc='center'
)

le_do = [round(d_do[i]["evaluation"]["t=-1"]["macro"]["mean"][:,1:].mean().item()) 
        for i in l_item]       
le_doe = [round(d_doe[i]["evaluation"]["t=-1"]["macro"]["mean"][:,1:].mean().item()) 
        for i in l_item]
print(le_do, le_doe)


count_do=0
count_doe=0
doe_best=0

for i in range(3):
    for j in range(2):
        for ll_item in l_item:
            for g,gom in results2.items():
                if gom[ll_item][i][j]>d_do[ll_item]["fitness"]["max"]["t=-1"]["macro"]["mean"][i][j+1]:
                    count_do+=1
                    print(i,j,ll_item,g)
                if gom[ll_item][i][j]>d_doe[ll_item]["fitness"]["max"]["t=-1"]["macro"]["mean"][i][j+1]:
                    count_doe+=1
                    print(i,j,ll_item,g)
                if d_do[ll_item]["fitness"]["max"]["t=-1"]["macro"]["mean"][i][j+1]>d_doe[ll_item]["fitness"]["max"]["t=-1"]["macro"]["mean"][i][j+1]:
                    doe_best+=1
