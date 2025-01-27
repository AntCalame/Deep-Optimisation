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

pop_size=1000
batch_size=pop_size

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
import data_to_store as dts

os.chdir(parent2_directory)

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

nb_item = 100
problem = MKP("COProblems/mkp/problems"+str(dim)+"d-"+str(nb_item)+".txt",
              "COProblems/mkp/fitnesses"+str(dim)+"d-"+str(nb_item)+".txt", idx_prb, device)

model = DOAE(problem_size, dropout_prob, device)
handler = OptimAEHandler(model, problem, device)

population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)

total_eval = 0
depth = 0

store = {"fitness" : [],
         "population" : [],
         "evaluation" : [],
         "time" : [],
         "transition_loss": []}
store_model = []

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
    
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))
    
    store_model += [copy.deepcopy(model)]
    store["population"] += [population.clone().detach()[None,...]]
    store["fitness"] += [fitnesses.clone().detach()[None,...]]
    store["time"] += [time.time()-t0]
    store["evaluation"] += [total_eval]
    store["transition_loss"] += [transition_loss]
    
save = dts.analyse(store,
            problem.max_fitness,
            store_model,
            nb_iteration,
            max_depth)

if "l1" in save["transition_loss"].keys():
    save["transition_loss"]["l1"] *= l1_coef

torch.save(save, "data_train/"+str(number_experiment)+'.pt')
torch.save([model.state_dict() for model in store_model],
           "model/"+str(number_experiment)+'.pth')
