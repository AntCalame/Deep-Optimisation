import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import seaborn as sns
import os

plt.rcParams.update({'font.size': 16})

SSS=r""

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

base_train = torch.load("data/exp_transfer.pt")
base_train = base_train["1"]
base_test = torch.load("data/exp_transfer_test.pt")
base_test = base_test["1"]

batch_train = torch.load("data/exp4.pt")
batch_train = batch_train["64"]
batch_test = torch.load("data/exp4_test.pt")
batch_test = batch_test["64"]

rehearsal_train = torch.load("data/exp5.pt")
rehearsal_train = rehearsal_train["0.4"]
rehearsal_test = torch.load("data/exp5_test.pt")
rehearsal_test = rehearsal_test["0.4"]

all_train = torch.load("data/exp_tuning_all_n1-n2.pt")
all_train = all_train["70000"]
all_test = torch.load("data/exp_tuning_all_n1-n2_test.pt")
all_test = all_test["70000"]

replace_train = torch.load("data/exp_tuning_replace_n1-n2.pt")
replace_train = replace_train["70000"]
replace_test = torch.load("data/exp_tuning_replace_n1-n2_test.pt")
replace_test = replace_test["70000"]

dim=["5","10","30"]
tight=["0.25","0.50","0.75"]

# fr per problem
# paper, base, 64, 0.4, all, replace

L_train=[base_train,batch_train,rehearsal_train,all_train,replace_train]
L_test=[base_test,batch_test,rehearsal_test,all_test,replace_test]
L_name=["Base Algorithm", "Batch: 64", "Rehearsal: 0.4", "All-EWC: 70000", "Replace-EWC: 70000"]


fig, ax = plt.subplots(figsize=(20, 7))
ax.axis('tight')
ax.axis('off')

lr=[0.3,0.63,0.65,
    "-","-","-",
    0.05,0.07,0.11]
        
table_do = [[dim[i]+"-"+tight[j]]+[lr[i+3*j]]+
             [round((1-d["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3) for d in L_train]
            for i in range(3) for j in range(3)]
table_do += [["total"]+["-"]+
             [round((1-d["fitness"]["max"]["t=all"]["global"]["mean"][-1].item())*100,3) for d in L_train]]
        
table = plt.table(
    cellText=table_do,
    colLabels=["Problem Type"]+["Results Extracted"]+L_name,
    loc='center',
    cellLoc='center', fontsize=16
)
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
for _, cell in table.get_celld().items():
    cell.set_height(0.08)
table.auto_set_font_size(False)
table.set_fontsize(20)
plt.title("Comparison of Fitness Ratio per problem type (\u2193)")
plt.tight_layout()
plt.savefig(SSS+'\\pp.pdf', format='pdf', bbox_inches='tight')
plt.show()
        
# paper, base, 64, 0.4, all, replace
# fr, sr, fr test, portability, final loss
fig, ax = plt.subplots(figsize=(20, 5))
ax.axis('tight')
ax.axis('off')
data=[]
for i in range(5):
    d_train = L_train[i]
    d_test = L_test[i]
    
    data+=[[round((1-d_train["fitness"]["max"]["t=-1"]["global"]["mean"].item())*100,3),
            round(d_train["fitness"]["max"]["t=-1"]["global"]["count1"].item(),3),
            round(d_train["time"]["t=-1"]["global"]["mean"].item()),
            round(d_train["evaluation"]["t=-1"]["global"]["mean"].item()),
            round(d_train["loss"]["global"]["mean"][-1][-1].item(),3),
            round((1-d_test["fitness"]["max"]["t=all"]["global"]["mean"][-1][-1].item())*100,3),
            round((d_test["fitness"]["max"]["t=all"]["global"]["mean"][-1][-1]/d_train["fitness"]["max"]["t=-1"]["global"]["mean"]*100).item(),3)
            ]]

metric=["Fitness Ratio (\u2193)","Success Rate (\u2191)", "Time (\u2193)", "Evaluations (\u2193)", "Final Loss (\u2193)","Fitness Ratio Test (\u2193)","Portability (%) (\u2191)"]
table_do = [[metric[i]]+[data[j][i] for j in range(5)] for i in range(7)]
table = plt.table(
    cellText=table_do,
    colLabels=["Metric"]+L_name,
    loc='center',
    cellLoc='center'
)
for key, cell in table.get_celld().items():
    if key[0] == 0 or key[1] == 0:  # If it's in the first row or first column
        cell.set_linewidth(2)  # Thicker line
    else:
        cell.set_linewidth(1)
for _, cell in table.get_celld().items():
    cell.set_height(0.1)
table.auto_set_font_size(False)
table.set_fontsize(17)
plt.title("Most important metrics for all models")
plt.savefig(SSS+'\\imp.pdf', format='pdf', bbox_inches='tight')
plt.show()

