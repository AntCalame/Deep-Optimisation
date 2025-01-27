import sys
import os
import time
import copy
import torch
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#--------------------------------
#---------manage path------------
#--------------------------------
current_directory = os.path.dirname(os.path.realpath(__file__))

parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

parent2_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
sys.path.append(parent2_directory)

path_COProblems = os.path.join(parent2_directory, "COProblems")
sys.path.append(path_COProblems)

path_mkp = os.path.join(path_COProblems, "mkp")
sys.path.append(path_mkp)

from COProblems.MKP import MKP

os.chdir(parent2_directory)

do_ = torch.load("HPC/data/exp_paper_do_sup1.pt")
do_ewc = torch.load("HPC/data/exp_paper_doewc_sup1.pt")

items=["100","250","500"]
dimension=["5","10","30"]

device="cpu"
device = torch.device(device)

L_do = [do_,do_ewc]

for do in L_do:
    for k in do.keys():
        for i in do[k].keys():
            for j in do[k][i].keys():
                for l in do[k][i][j].keys():
                    dim=dimension[int(i)]
                    nb_item=int(k)
                    idx_prb=10*int(j)+int(l)
                    problem = MKP("COProblems/mkp/problems"+str(dim)+"d-"+str(nb_item)+".txt",
                                  "COProblems/mkp/fitnesses"+str(dim)+"d-"+str(nb_item)+".txt",
                                  idx_prb, device)
                    
                    for z in do[k][i][j][l]:
                        if torch.any(problem.fitness(z[1])>problem.max_fitness):
                            #print(nb_item,
                            #      dim,
                            #      idx_prb)
                            #print(torch.max(problem.fitness(z[1])),problem.max_fitness)
                            if nb_item!=100:
                                print(torch.max(problem.fitness(z[1])),problem.max_fitness)
                        
                

