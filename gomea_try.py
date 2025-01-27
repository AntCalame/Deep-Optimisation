import gomea
from COProblems.MKP import MKP
import matplotlib.pyplot as plt
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
device = torch.device(device)

#5-10-30
dim=5
#100-250-500
nb_item=500

problem = MKP(f"COProblems\\mkp\\problems{dim}d-{nb_item}.txt",
              f"COProblems\\mkp\\fitnesses{dim}d-{nb_item}.txt",
              10, device)

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



# Working
#lm = gomea.linkage.Univariate()
#lm = gomea.linkage.Full()
lm = gomea.linkage.LinkageTree(similarity_measure = b'MI', filtered = True)

"""
todo:
    linkagetree : 4 = 2 sim * 2 filetered
    univariate : 1
    full : 1
    
"""


fd = CustomMKP(nb_item,value_to_reach=problem.max_fitness)

dgom  = gomea.DiscreteGOMEA(fitness=fd,
                            linkage_model=lm,
                            max_number_of_evaluations=10000)#100000000)
result = dgom.run()
a = result.printAllStatistics()


plt.grid()
plt.xlabel('Number of evaluations')
plt.ylabel('Best objective value')
plt.plot(result['evaluations'],result['best_obj_val'])
plt.show()