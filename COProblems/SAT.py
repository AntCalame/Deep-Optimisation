import torch

from COProblems import SAT_populate_function as sat
from COProblems.OptimisationProblem import OptimisationProblem

class SAT(OptimisationProblem):
    """
    Class to implement the Set Covering Problem.
    """
    def __init__(self, idx: int, device: torch.device):
        super().__init__(device)
        self.W, self.C = sat.SATpopulate(idx)
        
        self.C = self.C.to(dtype=torch.float32, device=device)
        self.W = self.W.to(dtype=torch.float32, device=device)
        
        self.max_fitness =  sat.SATFitness(idx) 
        print("Max possible fitness for this instance: {}".format(self.max_fitness))
        
    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        valid_clause = torch.any((x[...,None] * (self.C.T)[None,...])==1,1) 
        fitnesses = torch.sum(torch.where(valid_clause,self.W[None,...],0),1)
        return fitnesses
    
    def is_valid(self, x: torch.Tensor) -> torch.Tensor:
        return True
    
    def random_solution(self, pop_size: int) -> torch.Tensor:
        return torch.randint(0,2,(pop_size, self.C.shape[1]), device=self.device, dtype=torch.float32)

    def repair(self, s: torch.Tensor) -> torch.Tensor:
        return s