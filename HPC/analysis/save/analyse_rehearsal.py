import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import seaborn as sns
import os
import numpy as np

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

data = torch.load("data/exp5.pt")
SSS=r""

class Analysis:
    def __init__(self,data):
            
        self.D=data
        self.dim = ["5","10","30"]
        self.tight = ["0.25","0.5","0.75"]
    
    def analysis(self):
        plt.rcParams.update({'font.size': 16})

        fig, _ = plt.subplots(figsize=(8, 5.3))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(1,21),
                        d["fitness"]["max"]["t=all"]["global"]["mean"],
                     label=n)
        plt.legend(title="Rehearsal: ",ncol=2)
        plt.xlabel("Transition")
        plt.ylabel("Fitness Ratio")
        plt.xticks(torch.arange(1,21))
        plt.grid()
        plt.title("Comparison of Fitness Ratio (Rehearsal)")
        #plt.savefig(SSS+'\\fr.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(8, 5))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(1,21),
                        d["fitness"]["max"]["t=all"]["global"]["count1"],
                     label=n)
        plt.legend(title="Rehearsal: ",ncol=2)
        plt.xlabel("Transition")
        plt.ylabel("Success Rate")
        plt.xticks(torch.arange(1,21))
        plt.grid()
        plt.title("Comparison of Success Rate (Rehearsal)")
        #plt.savefig(SSS+'\\sr.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(8, 5.3))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(10,21),
                        d["fitness"]["max"]["t=all"]["global"]["mean"][9:],
                     label=n)
        plt.legend(title="Rehearsal: ",ncol=2)
        plt.xlabel("Transition")
        plt.xticks(torch.arange(10,21))
        plt.ylabel("Fitness Ratio")
        plt.grid()
        plt.title("Comparison of Fitness Ratio (Rehearsal) (zoomed in)")
        plt.tight_layout()
        #plt.savefig(SSS+'\\frz.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(8, 5))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(10,21),
                        d["fitness"]["max"]["t=all"]["global"]["count1"][9:],
                     label=n)
        plt.legend(title="Rehearsal: ",ncol=2)
        plt.xlabel("Transition")
        plt.xticks(torch.arange(10,21))
        plt.ylabel("Success Rate")
        plt.grid()
        plt.title("Comparison of Success Rate (Rehearsal) (zoomed in)")
        #plt.savefig(SSS+'\\srz.pdf', format='pdf', bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        
        
        n="0.4"
        d=self.D[n]
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
        plt.title("Detailed loss per component: "+str(n))
        plt.grid()
        #plt.savefig(SSS+'\\l-d-0.4.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(d["loss"]["global"]["mean"],
                    xticklabels=range(1,21),
                    yticklabels=range(1,21),
                    cbar_kws={'label': 'Loss: MSE'})
        plt.xlabel("Transition: Population")
        plt.ylabel("Transition: Model")
        plt.title("Loss per Model per Population: "+str(n))
        #plt.savefig(SSS+'\\l-pp-0.4.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(d["loss"]["global"]["mean"]/(d["loss"]["global"]["mean"].mean(axis=1,keepdim=True)),
                    xticklabels=range(1,21),
                    yticklabels=range(1,21),
                    cbar_kws={'label': 'Loss: MSE'})
        plt.xlabel("Transition: Population")
        plt.ylabel("Transition: Model")
        plt.title("Loss per Model per Population (M2): "+str(n))
        ##plt.savefig(SSS+'\\l-pp-70000.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        fig, _ = plt.subplots(figsize=(7, 5))
        sns.heatmap(d["loss"]["global"]["mean"][1:],
                    xticklabels=range(1,21),
                    yticklabels=range(2,21),
                    cbar_kws={'label': 'Loss: MSE'})
        plt.xlabel("Transition: Population")
        plt.ylabel("Transition: Model")
        plt.title("Loss per Model per Population: "+str(n))
        #plt.savefig(SSS+'\\l-pp-0.42.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        Key = ["0.1","0.2","0.3","0.4","0.5"]
        
        # Table DO
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.axis('tight')
        ax.axis('off')
        
        table_do = [[self.dim[i]+"-"+self.tight[j]]+
                     [round((1-self.D[k]["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item())*100,3) for k in Key]
                    for i in range(3) for j in range(3)]
        
        table_do += [["total"]+
                     [round((1-self.D[k]["fitness"]["max"]["t=all"]["global"]["mean"][-1].item())*100,3) for k in Key]]
        
        table = plt.table(
            cellText=table_do,
            colLabels=["Problem Type"]+Key,
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
            cell.set_height(0.08)
        plt.tight_layout()
        plt.title("Comparison of Fitness Ratio per problem type (\u2193)")
        #plt.savefig(SSS+'\\table-fr.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
        # Table DO
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.axis('tight')
        ax.axis('off')
        
        table_do = [[self.dim[i]+"-"+self.tight[j]]+
                     [round(self.D[k]["fitness"]["max"]["t=all"]["macro"]["count1"][i][j][-1].item(),5) for k in Key] 
                    for i in range(3) for j in range(3)]
        table_do += [["total"]+
                     [round(self.D[k]["fitness"]["max"]["t=all"]["global"]["count1"][-1].item(),2) for k in Key]]
        
        table = plt.table(
            cellText=table_do,
            colLabels=["Problem Type"]+Key,
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
            cell.set_height(0.08)
        plt.title("Comparison of Success Rate per problem type (\u2191)")
        plt.tight_layout()
        #plt.savefig(SSS+'\\table-sr.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        
A = Analysis(data)
A.analysis()
            