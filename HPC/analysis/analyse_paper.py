import os
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

do = torch.load("data/exp_paper_do.pt")
do_test = torch.load("data/exp_paper_do_test.pt")
do_retrain = torch.load("data/exp_paper_do_retrain.pt")

ewc = torch.load("data/exp_paper_doewc.pt")
ewc_test = torch.load("data/exp_paper_doewc_test.pt")
ewc_retrain = torch.load("data/exp_paper_doewc_retrain.pt")

raw_gomea = torch.load("data/ex_paper_gomea.pt")

plt.rcParams.update({'font.size': 16})

tight=["0.25","0.50","0.75"]
dim=["5","10","30"]
item=["100","250","500"]

colors = ["red","blue","green","pink","orange","purple"]


gomea = {}
for i in raw_gomea.keys():
    if i[-1]!="0":
        if not i[-1] in gomea.keys():
            gomea[i[-1]]={}
        gomea[i[-1]][i[0:3]]=raw_gomea[i]
    
map_gomea = {#"0" : "Univariate",
            "2" : "MI-NF",
            "3" : "MI-F",
            "4" : "NMI-NF",
            "5" : "NMI-F"}

for  k in range(3):
    fig, axes = plt.subplots(3, 3, figsize=(16, 6))
    
    for i in range(3):
        for j in range(3):
            axes[i][j].boxplot(do[item[k]]["fitness"]["max"]["t=-1"]["global"]["distribution"][i][j].reshape(-1),
                               positions=[2], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor=colors[0], color=colors[0], alpha=0.5),
                            medianprops=dict(color='black'),
                            whiskerprops=dict(color=colors[0]),
                            capprops=dict(color=colors[0]),
                            vert=False)
            
            
            axes[i][j].boxplot(ewc[item[k]]["fitness"]["max"]["t=-1"]["global"]["distribution"][i][j].reshape(-1),
                               positions=[1], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor=colors[1], color=colors[1], alpha=0.5),
                            medianprops=dict(color='black'),
                            whiskerprops=dict(color=colors[1]),
                            capprops=dict(color=colors[1]),
                            vert=False)
            
            if j!=0:
                for zz,z in enumerate(map_gomea.keys()):
                    axes[i][j].boxplot(gomea[z][item[k]]["fitness"]["max"]["global"]["distribution"][i][j-1].reshape(-1),
                                       positions=[3+zz], widths=0.4, patch_artist=True,
                                    boxprops=dict(facecolor=colors[2+zz], color=colors[2+zz], alpha=0.5),
                                    medianprops=dict(color='black'),
                                    whiskerprops=dict(color=colors[2+zz]),
                                    capprops=dict(color=colors[2+zz]),
                                    vert=False)
            

            axes[i][j].set_yticks([])
            axes[i][j].xaxis.set_major_locator(MaxNLocator(nbins=3))
            
            if j==0:
                axes[i][j].set_ylabel(dim[i])
            if i==2:
                axes[i][j].set_xlabel(tight[j])


    fig.suptitle(f"Fitness Ratios Boxplot per Problem Type ({item[k]} items)",fontsize=22)
    blue_patch = mpatches.Patch(color='blue', label='Replace-EWC', alpha=0.5)
    red_patch = mpatches.Patch(color='red', label='Base Algorithm', alpha=0.5)
    patchs = [blue_patch,red_patch]
    
    for p,(c,alg )in enumerate(map_gomea.items()):
        patchs += [mpatches.Patch(color=colors[2+p], label="Gomea "+alg, alpha=0.5)]
    
    plt.tight_layout()
    fig.supxlabel("Tightness",y=-0.01)
    fig.supylabel("Dimension",x=-0.01)
    axes[0][0].legend(handles=patchs, loc='upper left', bbox_to_anchor=(0, 1.4), 
               ncol=6, frameon=False)
    #plt.savefig(SSS+'\\bx.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    

plt.plot(1-do["100"]["fitness"]["max"]["t=all"]["global"]["mean"],
         label="do",color="cornflowerblue")
plt.plot(1-do_test["100"]["fitness"]["max"]["t=all"]["global"]["mean"][:,-1],
         label="do_test",color="lightskyblue")
plt.plot(1-do_retrain["100"]["fitness"]["max"]["t=all"]["global"]["mean"][:,-1],
         label="do_ewc",color="blue")

plt.plot(1-ewc["100"]["fitness"]["max"]["t=all"]["global"]["mean"],
         label="ewc",color="red")
plt.plot(1-ewc_test["100"]["fitness"]["max"]["t=all"]["global"]["mean"][:,-1],
         label="ewc_test", color="orange")
plt.plot(1-ewc_retrain["100"]["fitness"]["max"]["t=all"]["global"]["mean"][:,-1],
         label="ewc_retrain", color="firebrick")

plt.yscale('log')
plt.xlabel("transitions")
plt.ylabel("1-fitness ratios")
plt.title("Comparison Test/Retrain")
plt.legend(ncol=2)
plt.grid()
plt.show()
