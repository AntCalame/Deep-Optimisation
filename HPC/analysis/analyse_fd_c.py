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

data = torch.load("data/exp_furtherdim_classic.pt")
SSS=r""
class Analysis:
    def __init__(self,data):
        
        # 0-1 hill climbing
        # 0-2 upgraded
        D={}
        for i,j in data.items():
            if i=="500":
                self.D = j
            
        self.label = "EWC: 250 items"
        self.n = "EWC"
        
        self.dim = ["5","10","30"]
        self.tight = ["0.25","0.5","0.75"]
    
    def analysis(self):
        
        # First plot
        plt.plot(torch.arange(1,21),
                    self.D["fitness"]["max"]["t=all"]["global"]["count1"],
                    label=self.n,
                    c='black',
                    linewidth=2)
        plt.xticks(torch.arange(1,21))
        plt.legend()
        plt.xlabel("Transition")
        plt.ylabel("Success Rate")
        plt.title("Success Rate: " + str(self.label))
        plt.grid()
        plt.show()
        
        plt.plot(torch.arange(1,21),
                    self.D["fitness"]["max"]["t=all"]["global"]["mean"],
                    label=self.n,
                    c='black',
                    linewidth=2)
        plt.xticks(torch.arange(1,21))
        plt.legend()
        plt.xlabel("Transition")
        plt.ylabel("Fitness Ratio")
        plt.title("Fitness Ratio: " + str(self.label))
        plt.grid()
        plt.show()
        
        plt.rcParams.update({'font.size': 16})
        
        # Create a figure and a 2x1 grid of subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 5),sharey=True)  # (2, 1) specifies 2 rows and 1 column
        
        # First plot
        axs[0].plot(torch.arange(1,21),
                    self.D["fitness"]["max"]["t=all"]["global"]["count1"],
                    label=self.n,
                    c='black',
                    linewidth=2)
        axs[0].set_xticks(torch.arange(1,21))
        axs[0].legend()
        axs[0].set_xlabel("Transition")
        axs[0].set_ylabel("Success Rate")
        axs[0].set_title("Success Rate: " + str(self.label))
        axs[0].grid()
        
        # Second plot
        for i in range(3):
            for j in range(3):
                color = (j+2)/4 * (np.arange(3)==i)
                color += 0.3*(j+2)/4 * ((np.arange(3)-i)%3==2)
                axs[1].plot(torch.arange(1,21),
                            self.D["fitness"]["max"]["t=all"]["macro"]["count1"][i][j],
                            label=self.dim[i] + "-" + self.tight[j],
                            c=color,
                            alpha=0.5,
                            linewidth=2)  # Color for the lines
        axs[1].legend(title="Type of Problem: ", ncol=3,fontsize=13)
        axs[1].set_xticks(torch.arange(1,21))
        axs[1].set_xlabel("Transition")
        axs[1].set_title("Success Rate for every problem type")
        axs[1].grid()
        
        plt.tight_layout()
        #plt.savefig(SSS+'\\sr.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        # Create a figure and a 2x1 grid of subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 5),sharey=True)  # (2, 1) specifies 2 rows and 1 column
        for i in range(3):
            for j in range(3):
                color = (j+2)/4 * (np.arange(3)==i)
                color += 0.3*(j+2)/4 * ((np.arange(3)-i)%3==2)
                axs[0].plot(torch.arange(1,21),
                            self.D["fitness"]["max"]["t=all"]["macro"]["mean"][i][j],
                         label=self.dim[i]+"-"+self.tight[j],
                         color=color,
                         alpha=0.5,
                         linewidth=2)
        axs[0].legend(title="Type of Problem: ",ncol=3,fontsize=13)
        axs[0].set_xlabel("Transition")
        axs[0].set_ylabel("Fitness Ratio")
        axs[0].set_xticks(torch.arange(1,21))
        axs[0].set_title("Fitness Ratio")
        axs[0].grid()
        
        for i in range(3):
            for j in range(3):
                color = (j+2)/4 * (np.arange(3)==i)
                color += 0.3*(j+2)/4 * ((np.arange(3)-i)%3==2)
                axs[1].plot(self.D["fitness"]["max"]["t=all"]["macro"]["count1"][i][j],
                         self.D["fitness"]["max"]["t=all"]["macro"]["mean"][i][j],
                         label=self.dim[i]+"-"+self.tight[j],
                         color=color,
                         marker='o',
                         alpha=0.5,
                         linewidth=2)
        axs[1].legend(title="Type of Problem: ",ncol=3,fontsize=13)
        axs[1].set_xlabel("Success Rate")
        axs[1].set_title("Fitness Ratio function of Success Rate")
        axs[1].grid()
        plt.tight_layout()
        #plt.savefig(SSS+'\\fr-sr.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        # Table DO
        fig, ax = plt.subplots(figsize=(11, 3))
        ax.axis('tight')
        ax.axis('off')
        lr=[0.3,0.63,0.65,
            "-","-","-",
            0.05,0.07,0.11]
        
        
        table_do = [[self.dim[i]+"-"+self.tight[j],
                     round((1-self.D["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,2),
                     lr[i+3*j]] 
                    for i in range(3) for j in range(3)]
        
        table = plt.table(
            cellText=table_do,
            colLabels=["Problem Type",
                       "Base Algorithm",
                       "Results Extracted"],
            loc='center',
            cellLoc='center',
            colWidths = [0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.2)

        for i in range(3):
            for j in range(3):
                color = (j+2)/4 * (np.arange(3)==i)
                color += 0.3*(j+2)/4 * ((np.arange(3)-i)%3==2)

                color = np.append(color, 0.5)
                table[(1+3*i+j, 0)].set_facecolor(color)
                table[(1+3*i+j, 1)].set_facecolor(color)
                table[(1+3*i+j, 2)].set_facecolor(color)
        plt.title("Comparison of Fitness Ratio per Problem Type (\u2193)")
        for key, cell in table.get_celld().items():
            cell.set_height(0.1)
        plt.tight_layout()
        #plt.savefig(SSS+'\\table-comparison.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        for i in range(2,-1,-1):
            color = (np.arange(3)==i)*np.ones(3)
            color += 0.3 * ((np.arange(3)-i)%3==2)
            
            plt.hist(self.D["fitness"]["max"]["t=-1"]["global"]["distribution"][i].reshape(-1),
                     label=self.dim[i],
                     color=color,
                     alpha=0.5)
        plt.legend(title="Dimension: ",ncol=3,fontsize=13)
        plt.xlabel("Fitness Ratio")
        plt.title("Distribution of Fitness Ratio (Dimension)")
        plt.grid()
        plt.tight_layout()
        #plt.savefig(SSS+'\\per-dim.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        

        for j in range(3):
            color = (np.arange(3)==j)*np.ones(3)
            color += 0.3 * ((np.arange(3)-j)%3==2)
            plt.hist(self.D["fitness"]["max"]["t=-1"]["global"]["distribution"][:,j].reshape(-1),
                     label=self.tight[j],
                     color=color,
                     alpha=0.5)
        plt.legend(title="Tightness ratio: ",ncol=3,fontsize=13)
        plt.xlabel("Fitness Ratio")
        plt.title("Distribution of Fitness Ratio (Tightness)")
        plt.grid()
        plt.tight_layout()
        #plt.savefig(SSS+'\\per-tightness.pdf', format='pdf', bbox_inches='tight')
        plt.show()

        
        L_loss=[]
        L_label=[]
        
        for i,j in self.D["transition_loss"].items():
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
        plt.xlabel("Transition")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend(title="Component:")
        plt.title("Detailed loss per component")
        #plt.savefig(SSS+'\\loss-detail.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(self.D["loss"]["global"]["mean"],
                    xticklabels=range(1,21),
                    yticklabels=range(1,21),
                    cbar_kws={'label': 'Loss: MSE'})
        plt.xlabel("Transition: Population")
        plt.ylabel("Transition: Model")
        plt.title("Loss per Model per Population (M1)")
        #plt.savefig(SSS+'\\loss-perm-perp.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(self.D["loss"]["global"]["mean"]/(self.D["loss"]["global"]["mean"].mean(axis=1,keepdim=True)),
                    xticklabels=range(1,21),
                    yticklabels=range(1,21),
                    cbar_kws={'label': 'Loss: MSE'})
        plt.xlabel("Transition: Population")
        plt.ylabel("Transition: Model")
        plt.title("Loss per Model per Population")
        #plt.savefig(SSS+'\\loss-perm-perp.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(self.D["correlation"]["global"]["mean"][1:],
                    xticklabels=range(1,21),
                    yticklabels=range(2,21),
                    cbar_kws={'label': 'Correlation'},
                    cmap='viridis')
        plt.xlabel("Transition: Population")
        plt.ylabel("Transition: Model")
        plt.title("Correlation (Loss-Fitness)")
        #plt.savefig(SSS+'\\correlation.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(self.D["dl1"]["global"]["mean"],
                    xticklabels=range(1,21),
                    yticklabels=range(1,21),
                    cbar_kws={'label': 'Distance'},
                    cmap='Spectral') 
        plt.xlabel("Transition: Model")
        plt.ylabel("Transition: Model")
        plt.title("Layer 1 Distances: Base Algorithm")
        #plt.savefig(SSS+'\\dl.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        
A = Analysis(data)
A.analysis()
            