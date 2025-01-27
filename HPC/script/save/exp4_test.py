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
batch_size = int(sys.argv[3])

pop_size=1000

number_experiment = int(number_experiment0//4)
idx_models = int(number_experiment0-4*number_experiment)

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
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler

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
l1_coef = 0.0001
l2_coef = 0.0001
lr = 0.002
compression_ratio = 0.8
hidden_size = problem_size

problem = MKP("COProblems/mkp/problems"+str(dim)+"d.txt", 
              "COProblems/mkp/fitnesses"+str(dim)+"d.txt", idx_prb, device)

loaded_models = [DOAE(problem_size, dropout_prob, device)for _ in range(len(loaded_state_dicts))]

for i,h in enumerate(loaded_models):
    hidden_size = problem_size
    
    for j in range(min(i+1,max_depth)):
        hidden_size = round(hidden_size*compression_ratio)
        h.transition(hidden_size)
        
for model, state_dict in zip(loaded_models, loaded_state_dicts):
    model.load_state_dict(state_dict)

nb_iteration = 20

Save = {}
for idx_model in range(idx_models*5,(idx_models+1)*5):
    total_eval = 0
    
    store = {"fitness" : [],
            "eval" : [],
            "time" : []}
    
    model = loaded_models[idx_model]
    handler = OptimAEHandler(model, problem, device)
    
    population, fitnesses = handler.generate_population(pop_size)
    population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)
    
    t0 = time.time()
    for ii in range(nb_iteration):
        print("Transition: ",ii," /",nb_iteration)
        
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