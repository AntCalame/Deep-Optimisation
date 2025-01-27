import sys
import os
import time
import copy
import torch
import numpy as np

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

#--------------------------------
#---------get var----------------
#--------------------------------

number_experiment = int(sys.argv[1])
size_experiment = int(sys.argv[2])
nb_item = int(sys.argv[3])
configuration = int(sys.argv[4])

l_dim = [5,10,30]
idx_dim = int(np.floor(number_experiment/(20*size_experiment)))
dim = l_dim[idx_dim]

idx_prb = int(np.floor((number_experiment-idx_dim*(20*size_experiment))/size_experiment))+10

print("Number: ",number_experiment)
print("Dim: ",dim)
print("Problem: ",idx_prb)
print()

#--------------------------------
#-------------------------------
#--------------------------------

import gomea
from COProblems.MKP import MKP
import numpy as np
import torch

os.chdir(parent2_directory)

device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
device = torch.device(device)

problem = MKP("COProblems/mkp/problems"+str(dim)+"d-"+str(nb_item)+".txt",
              "COProblems/mkp/fitnesses"+str(dim)+"d-"+str(nb_item)+".txt",
              idx_prb, device)

c = problem.c.numpy()
b = problem.b.numpy()
a = problem.A.numpy()

class CustomMKP(gomea.fitness.BBOFitnessFunctionDiscrete):
    def objective_function( self, objective_index, variables ):
        
        valid = np.all(variables@a.T<b)

        fitness = variables@c
        
        
        if valid:
            return fitness
    
        else: 
            return 0.

if configuration==0:
    lm = gomea.linkage.Univariate()

elif configuration==1:
    lm = gomea.linkage.Full()
    
elif configuration==2:
    lm = gomea.linkage.LinkageTree(similarity_measure = b'MI', filtered = False)

elif configuration==3:
    lm = gomea.linkage.LinkageTree(similarity_measure = b'MI', filtered = True)
    
elif configuration==4:
    lm = gomea.linkage.LinkageTree(similarity_measure = b'NMI', filtered = False)
    
elif configuration==5:
    lm = gomea.linkage.LinkageTree(similarity_measure = b'NMI', filtered = True)

fd = CustomMKP(nb_item,value_to_reach=problem.max_fitness)

dgom  = gomea.DiscreteGOMEA(fitness=fd,
                            linkage_model=lm,
                            max_number_of_evaluations=100000000)
result = dgom.run()
result.printAllStatistics()

save = result.getGenerationalStatistics()
for i in range(len(save['best_obj_val'])):
    save['best_obj_val'][i]/=problem.max_fitness
    

torch.save(result.getGenerationalStatistics(), "data_train/"+str(number_experiment)+'.pt')