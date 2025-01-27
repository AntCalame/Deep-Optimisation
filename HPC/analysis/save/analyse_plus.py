import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import numpy as np
import os

SSS=r""

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

base = torch.load("data/exp_transfer.pt")
base = base["1"]

replace = torch.load("data/exp_tuning_replace_n1-n2.pt")
replace = replace["70000"]

plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(3, 3, figsize=(16, 6))

tight=["0.25","0.50","0.75"]
dim=["5","10","30"]

for i in range(3):
    for j in range(3):
        axes[i][j].boxplot(replace["fitness"]["max"]["t=-1"]["global"]["distribution"][i][j].reshape(-1),
                           positions=[1], widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor='blue', color='blue', alpha=0.5),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='blue'),
                        capprops=dict(color='blue'),
                        vert=False)
        
        axes[i][j].boxplot(base["fitness"]["max"]["t=-1"]["global"]["distribution"][i][j].reshape(-1),
                           positions=[2], widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor='red', color='red', alpha=0.5),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='red'),
                        capprops=dict(color='red'),
                        vert=False)
        
        axes[i][j].set_yticks([])
        axes[i][j].xaxis.set_major_locator(MaxNLocator(nbins=3))
        
        if i==2:
            axes[i][j].set_xlabel(dim[j])
        if j==0:
            axes[i][j].set_ylabel(tight[i])


fig.suptitle("Fitness Ratios Boxplot per Problem Type",fontsize=22)
blue_patch = mpatches.Patch(color='blue', label='Replace-EWC', alpha=0.5)
red_patch = mpatches.Patch(color='red', label='Base Algorithm', alpha=0.5)
plt.tight_layout()
fig.supxlabel("Dimension",y=-0.01)
fig.supylabel("Tightness",x=-0.01)
axes[0][0].legend(handles=[red_patch,blue_patch], loc='upper left', bbox_to_anchor=(0, 1.4), 
           ncol=2, frameon=False)
plt.savefig(SSS+'\\bx.pdf', format='pdf', bbox_inches='tight')
plt.show()


fig, ax = plt.subplots(figsize=(20, 5))
ax.axis('tight')
ax.axis('off')
table_do = [[dim[i]+"-"+tight[j]]+
             [round((1-replace["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
              round((1-replace["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100-
                     (1-base["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3),
              round(((replace["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item()-
                     base["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())/
                     (1-base["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item()))*100,1),
              round((replace["fitness"]["max"]["t=all"]["macro"]["count1"][i][j][-1].item()),1),
              round((replace["fitness"]["max"]["t=all"]["macro"]["count1"][i][j][-1].item()-
                     base["fitness"]["max"]["t=all"]["macro"]["count1"][i][j][-1].item()),2)]
            for i in range(3) for j in range(3)]
table_do += [["total"]+
             [round((1-replace["fitness"]["max"]["t=all"]["global"]["mean"][-1].item())*100,3),
              round((1-replace["fitness"]["max"]["t=all"]["global"]["mean"][-1].item())*100-
                     (1-base["fitness"]["max"]["t=all"]["global"]["mean"][-1].item())*100,3),
              round(((replace["fitness"]["max"]["t=all"]["global"]["mean"][-1].item()-
                     base["fitness"]["max"]["t=all"]["global"]["mean"][-1].item())/
                     (1-base["fitness"]["max"]["t=all"]["global"]["mean"][-1].item()))*100,1),
              round((replace["fitness"]["max"]["t=all"]["global"]["count1"][-1].item()),2),
              round((replace["fitness"]["max"]["t=all"]["global"]["count1"][-1].item()-
                     base["fitness"]["max"]["t=all"]["global"]["count1"][-1].item()),2)]]

L_metric=["Fitness Ratio (\u2193)",
          "Comparison FR (\u2193)",
          "Improvement (%) (\u2191)",
          "Success Rate (\u2191)",
          "Comparison SR (\u2191)"]

table = plt.table(
    cellText=table_do,
    colLabels=["Problem Type"]+L_metric,
    loc='center',
    cellLoc='center'
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
for key, cell in table.get_celld().items():
    cell.set_height(0.09)
table.auto_set_font_size(False)
table.set_fontsize(17)
plt.title("Improvement of Deep Optimisation: Replace-EWC")
plt.savefig(SSS+'\\table-opti.pdf', format='pdf', bbox_inches='tight')

