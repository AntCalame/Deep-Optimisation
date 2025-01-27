import torch

from COProblems import SCP_populate_function as scp
from COProblems.OptimisationProblem import OptimisationProblem

class SCP(OptimisationProblem):
    """
    Class to implement the Set Covering Problem.
    """
    def __init__(self, idx: int, device: torch.device):
        super().__init__(device)
        self.C, self.A = scp.SCPpopulate(idx)
        self.C = self.C.to(dtype=torch.float32, device=device)
        self.A = self.A.to(dtype=torch.float32, device=device)
       
        self.correction = torch.sum(torch.where(self.C>0,self.C,0))
        
        self.max_fitness =  - scp.SCPFitness(idx) + self.correction
        print("Max possible fitness for this instance: {}".format(self.max_fitness))
        
    def fitness(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        return torch.where(self.is_valid(x), - x.matmul(self.C) + self.correction, 0)
    
    def is_valid(self, x: torch.Tensor) -> torch.Tensor:
        return (x.matmul(self.A.T) > 0).all(dim=1)
    
    def random_solution(self, pop_size: int) -> torch.Tensor:
        return torch.full((pop_size, self.C.shape[0]), 1, device=self.device, dtype=torch.float32)

    def repair(self, s: torch.Tensor) -> torch.Tensor:
        return s