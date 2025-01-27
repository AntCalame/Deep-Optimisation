import torch
import matplotlib.pyplot as plt

from COProblems.SCP import SCP
from Models.DOAE2 import DOAE_detail
from OptimAE import OptimAEHandler

# Highly recommended to keep as cpu for problems of size <= 100
device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
device = torch.device(device)

change_tolerance = 100
pop_size = 1000

problem = SCP(1, device)

problem_size = problem.C.size(0)

l1_coef = 0.00005
l2_coef = 0.0001
dropout_prob = 0.2
lr = 0.0002
compression_ratio = 0.8

model = DOAE_detail(problem_size, dropout_prob, device)
hidden_size = problem_size
handler = OptimAEHandler(model, problem, device)

population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)
handler.print_statistics(fitnesses)

total_eval = 0
max_depth = 6
depth = 0

while True:
    if depth < max_depth:
        hidden_size = round(hidden_size * compression_ratio)
        model.transition(hidden_size)
        depth += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
    print("Learning from population")
    # Learing with the entire population as a batch is technically not the best from a machine learning perspective,
    # but does not seem to have a massive impact on solution quality whilst also increasing learning speed significantly.
    z = handler.learn_from_population_detail(population, optimizer, l1_coef=l1_coef, batch_size=int(pop_size/10), epochs=50)
    
    for i in z.keys():
        if i=="l2":
            plt.plot(z[i]*l2_coef,label=i)
        elif i=="l1":
            plt.plot(z[i]*l1_coef,label=i)
        else:
            plt.plot(z[i],label=i)
    plt.legend()
    plt.show()
    
    print("Optimising population")
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance, encode=True, repair_solutions=False, deepest_only=False
    )
    handler.print_statistics(fitnesses)
    total_eval += evaluations
    print("Evaluations: {}".format(total_eval))
    
    # Uncomment lines below to print out best solution at every transition
    # best_i = torch.argmax(fitnesses)
    # print("Best solution - fitness = {}".format(fitnesses[best_i].item()))
    # print(population[best_i].tolist())
    if done:
        break
    



