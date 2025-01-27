import torch

from COProblems.MKP import MKP
from Models.DOAE_EWC import DOAE_EWC
from OptimAE_EWC import EWCOptimAEHandler

# Highly recommended to keep as cpu for problems of size <= 100
device = "cuda" if torch.cuda.is_available() else "cpu"
device="cpu"
print(device)
device = torch.device(device)

change_tolerance = 100
problem_size = 250
pop_size = 1000

problem = MKP("COProblems\\mkp\\problems30d-250.txt", "COProblems\\mkp\\fitnesses30d-250.txt", 11, device)

param = 70000
dropout_prob = 0.2
lr = 0.002
compression_ratio = 0.8

mode="replace"
model = DOAE_EWC(problem_size, dropout_prob, device, param, mode)
hidden_size = problem_size
handler = EWCOptimAEHandler(model, problem, device)

population, fitnesses = handler.generate_population(pop_size)
population, fitnesses, _, _ = handler.hillclimb(population, fitnesses, change_tolerance)
handler.print_statistics(fitnesses)

total_eval = 0
max_depth = 6
depth = 0

while depth < max_depth:
    if depth < max_depth:
        hidden_size = round(hidden_size * compression_ratio)
        model.transition(hidden_size)
        depth += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    
    print("Learning from population")
    # Learing with the entire population as a batch is technically not the best from a machine learning perspective,
    # but does not seem to have a massive impact on solution quality whilst also increasing learning speed significantly.
    handler.learn_from_population(population, optimizer, l1_coef=0, batch_size=pop_size)
    
    print("Optimising population")
    population, fitnesses, evaluations, done = handler.optimise_solutions(
        population, fitnesses, change_tolerance, encode=True, repair_solutions=True, deepest_only=False
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


