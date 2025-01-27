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

class Analysis:
    def __init__(self,data):
            
        self.D=data
        self.dim = ["5","10","30"]
        self.tight = ["0.25","0.5","0.75"]
    
    def analysis(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(1,21),
                        d["fitness"]["max"]["t=all"]["global"]["mean"],
                     label=n)
        plt.legend(title="Batch size: ")
        plt.xlabel("Transition")
        plt.ylabel("Fitness Ratio")
        plt.xticks(torch.arange(1,21))
        plt.grid()
        plt.title("Comparison of Fitness Ratio for different batch sizes")
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(1,21),
                        d["fitness"]["max"]["t=all"]["global"]["count1"],
                     label=n)
        plt.legend(title="Batch size: ")
        plt.xlabel("Transition")
        plt.ylabel("Success Rate")
        plt.xticks(torch.arange(1,21))
        plt.grid()
        plt.title("Comparison of Success Rate for different batch sizes")
        
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(10,21),
                        d["fitness"]["max"]["t=all"]["global"]["mean"][9:],
                     label=n)
        plt.legend(title="Batch size: ")
        plt.xlabel("Transition")
        plt.xticks(torch.arange(10,21))
        plt.ylabel("Fitness Ratio")
        plt.grid()
        plt.title("Comparison of Fitness Ratio for different batch sizes (zoomed in)")
        plt.tight_layout()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        for (n,d) in self.D.items():
            plt.plot(torch.arange(10,21),
                        d["fitness"]["max"]["t=all"]["global"]["count1"][9:],
                     label=n)
        plt.legend(title="Batch size: ")
        plt.xlabel("Transition")
        plt.xticks(torch.arange(10,21))
        plt.ylabel("Success Rate")
        plt.grid()
        plt.title("Comparison of Success Rate for different batch sizes (zoomed in)")
        
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 4))
        for (n,d) in self.D.items():
            plt.plot(d["fitness"]["max"]["t=all"]["global"]["count1"],
                     d["fitness"]["max"]["t=all"]["global"]["mean"],
                     marker='o',
                     label=n,
                     alpha=0.3)
        plt.legend(title="Batch size: ")
        plt.xlabel("Succes Rate")
        plt.ylabel("Fitness Ratio")
        plt.grid()
        plt.title("Fitness Ratio functio of Success Rate for different batch sizes")
        plt.tight_layout()
        plt.show()

        for (n,d) in self.D.items():
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
                
            for i in range(len(L_loss)):

                #plt.plot(torch.arange(0,j['global'].size(0)*j['global'].size(1))/j['global'].size(1),
                #         L_loss[i],
                #         label=L_label[i])

                plt.fill_between(torch.arange(0,j['global'].size(0)*j['global'].size(1))/j['global'].size(1),
                                 0,
                                 L_loss[i],
                                 label=L_label[i])
                
            plt.xticks(torch.arange(0,21))
            plt.xlabel("Succes Rate")
            plt.ylabel("Loss")
            plt.legend(title="Batch size: ")
            plt.title("Detailed loss during the optimisation process: "+str(n))
            plt.grid()
            plt.show()
        
        # compare loss
        for (n,d) in self.D.items():
            sns.heatmap(d["loss"]["global"]["mean"],
                        xticklabels=range(1,21),
                        yticklabels=range(1,21),
                        cbar_kws={'label': 'Loss: MSE'})
            plt.xlabel("Transition: Population")
            plt.ylabel("Transition: Model")
            plt.title("Loss per Model per Population: "+str(n))
            plt.show()
            
        # compare loss
        for (n,d) in self.D.items():
            sns.heatmap(d["correlation"]["global"]["mean"][1:],
                        xticklabels=range(1,21),
                        yticklabels=range(2,21),
                        cbar_kws={'label': 'Correlation'},
                        cmap='viridis')
            plt.xlabel("Transition: Population")
            plt.ylabel("Transition: Model")
            plt.title("Correlation between reconstruction loss and fitness: "+str(n))
            plt.show()
            
        # Table DO
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table_do = [[self.dim[i]+"-"+self.tight[j]]+
                     [round(d["fitness"]["max"]["t=all"]["macro"]["mean"][i][j][-1].item(),5) for n,d in self.D.items()]
                    for i in range(3) for j in range(3)]
        
        table_do += [["total"]+
                     [round(d["fitness"]["max"]["t=all"]["global"]["mean"][-1].item(),5) for n,d in self.D.items()]]
        
        table = plt.table(
            cellText=table_do,
            colLabels=["Problem (Dimension-Tightness)"]+[d for d in self.D.keys()],
            loc='center',
            cellLoc='center',
            colWidths = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)

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
        plt.title("Comparison of Fitness Ratio per problem type")
        plt.tight_layout()
        plt.show()
        
        # Table DO
        fig, ax = plt.subplots(figsize=(20, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table_do = [[self.dim[i]+"-"+self.tight[j]]+
                     [round(d["fitness"]["max"]["t=all"]["macro"]["count1"][i][j][-1].item(),5) for n,d in self.D.items()] 
                    for i in range(3) for j in range(3)]
        table_do += [["total"]+
                     [round(d["fitness"]["max"]["t=all"]["global"]["count1"][-1].item(),2) for n,d in self.D.items()]]
        
        table = plt.table(
            cellText=table_do,
            colLabels=["Problem (Dimension-Tightness)"]+[d for d in self.D.keys()],
            loc='center',
            cellLoc='center',
            colWidths = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.2)

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
        plt.title("Comparison of Success Rate per problem type")
        plt.tight_layout()
        plt.show()
        
        
A = Analysis(data)
A.analysis()
            