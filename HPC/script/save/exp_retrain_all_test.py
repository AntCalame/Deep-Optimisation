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

number_experiment0 = int(sys.argv[1])
size_experiment = int(sys.argv[2])
param = int(sys.argv[3])

pop_size=1000
    
batch_size=pop_size

number_experiment = int(number_experiment0//20)
idx_models = int(number_experiment0-20*number_experiment)

l_dim = [5,10,30]
idx_dim = int(np.floor(number_experiment/(30*size_experiment)))
dim = l_dim[idx_dim]

idx_prb = int(np.floor((number_experiment-idx_dim*(30*size_experiment))/size_experiment))

nb_iteration = 20

print("Number: ",number_experiment)
print("Dim: ",dim)
print("Problem: ",idx_prb)
print()

#--------------------------------
#-------------------------------
#--------------------------------

from COProblems.MKP import MKP
from Models.DOAE_EWC_Optimized import DOAE_EWC
from OptimAE_EWC_V2 import EWCOptimAEHandler

os.chdir(parent2_directory)

# Load the models' state_dicts
loaded_state_dicts = torch.load("model/"
                                +str(number_experiment)+'.pth')

problem_size = 100

device="cpu"
device = torch.device(device)
change_tolerance = 100

max_depth = 6
dropout_prob = 0.2
l1_coef = 0.
l2_coef = 0.
lr = 0.002
compression_ratio = 0.8
hidden_size = problem_size

idx_model = idx_models

nb_item = 100
problem = MKP("COProblems/mkp/problems"+str(dim)+"d-"+str(nb_item)+".txt",
              "COProblems/mkp/fitnesses"+str(dim)+"d-"+str(nb_item)+".txt", idx_prb, device)

mode="all"
loaded_models = [DOAE_EWC(problem_size, dropout_prob, device, param, mode) for _ in range(len(loaded_state_dicts))]
model = loaded_models[idx_model]

depth=min(max_depth,idx_model)
for j in range(min(idx_model+1,max_depth)):
    hidden_size = round(hidden_size*compression_ratio)
    model.transition(hidden_size)
    
for k,state_dict in enumerate(loaded_state_dicts):
    if k==idx_model:
        model.load_state_dict(state_dict)

if depth>=max_depth:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
nb_iteration = 20
Save = {}
total_eval = 0
store = {"fitness" : [],
        "eval" : [],
        "time" : []}

handler = EWCOptimAEHandler(model, problem, device)

population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)

t0 = time.time()
for ii in range(nb_iteration):
    print("Transition: ",ii," /",nb_iteration)
    
    if depth < max_depth:
        hidden_size = round(hidden_size * compression_ratio)
        model.transition(hidden_size)
        depth += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    transition_loss = handler.learn_from_population_detail(population, optimizer, l1_coef=l1_coef, batch_size=batch_size)
    
    
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance, encode=True, repair_solutions=True, deepest_only=False
    )

    total_eval += evaluations

    store["fitness"] += [fitnesses.clone().detach()[None,...]]
    store["time"] += [time.time()-t0]
    store["eval"] += [total_eval]

save = {"fit" : torch.cat(store["fitness"],0)/problem.max_fitness,
        "time" : torch.FloatTensor(store["time"]),
        "eval" : torch.FloatTensor(store["eval"])}
   
Save[str(idx_model)]=save

torch.save(Save, "data_test/"+str(number_experiment)+"-"+str(idx_models)+'.pt')
