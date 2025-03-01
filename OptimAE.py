import torch
from torch.utils.data import DataLoader, TensorDataset

from COProblems.OptimisationProblem import OptimisationProblem
from Models.DOAE import DOAE
from OptimHandler import OptimHandler

class OptimAEHandler(OptimHandler):
    """
    Describes the algorithm for carrying out DO with an AE model as specified in 
    "Deep Optimisation: Learning and Searching in Deep Representations of Combinatorial
    Optimisation Problems", Jamie Caldwell.
    """
    def __init__(self, model: DOAE, problem: OptimisationProblem, device: torch.device):
        """
        Constructor method for OptimAEHandler.

        Args:
            model: DO
                The central AE model used in Deep Optimisation.
            problem: OptimisationProblem
                The problem being solved.
            device: torch.device
                The device the model and problem are loaded onto.
        """
        super().__init__(model, problem, device)
    
    def learn_from_population(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              l1_coef: float = 0.0, batch_size: int = 16, epochs: int = 400, 
                              print_loss: bool = False) -> None:
        """
        Method to make the AE learn from the population of solutions.

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            optimizer: torch.optim.Optimizer
                The optimizer used to adjust the weights of the model.
            l1_coef: int
                The coefficient of the L1 term in the loss function.
            batch_size: int
                The batch size used during the learning process.
            epochs: int
                The number of epochs to train for.
            print_loss: bool
                If true, information regarding the reconstruction loss of the model will be 
                outputted.
        """
        total_recon = 0
        dataset = DataLoader(TensorDataset(solutions), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for i,x in enumerate(dataset):
                loss = self.model.learn_from_sample(x[0], optimizer, l1_coef)
                total_recon += loss["recon"]
            if print_loss:
                if (epoch+1) % 10 == 0:
                    print("Epoch {}/{} - Recon Loss = {}".format(
                        epoch+1,epochs,total_recon/(10*len(dataset))
                    ))
                    total_recon = 0
                    
    def learn_from_population_detail(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              l1_coef: float = 0.0, batch_size: int = 16, epochs: int = 400, 
                              print_loss: bool = False) -> float:
        """
        Method to make the AE learn from the population of solutions.

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            optimizer: torch.optim.Optimizer
                The optimizer used to adjust the weights of the model.
            l1_coef: int
                The coefficient of the L1 term in the loss function.
            batch_size: int
                The batch size used during the learning process.
            epochs: int
                The number of epochs to train for.
            print_loss: bool
                If true, information regarding the reconstruction loss of the model will be 
                outputted.
        """
        total_recon = 0
        dataset = DataLoader(TensorDataset(solutions), batch_size=batch_size, shuffle=True)
        detailed_loss = []
        for epoch in range(epochs):
            for i,x in enumerate(dataset):
                loss = self.model.learn_from_sample(x[0], optimizer, l1_coef)
                total_recon += loss["recon"]
                detailed_loss += [loss]
            if print_loss:
                if (epoch+1) % 10 == 0:
                    print("Epoch {}/{} - Recon Loss = {}".format(
                        epoch+1,epochs,total_recon/(10*len(dataset))
                    ))
                    total_recon = 0
        
        dict_loss = {}
        for i in detailed_loss[0].keys():
            specific_loss = []
            for j in detailed_loss:
                specific_loss += [j[i]]
            dict_loss[i]=torch.FloatTensor(specific_loss).detach()
        
        return dict_loss

    @torch.no_grad()
    def optimise_solutions(self, solutions: torch.Tensor, fitnesses: torch.Tensor,
                           change_tolerance : int, encode: bool=False,
                           repair_solutions: bool=False, deepest_only: bool=False) -> tuple[torch.Tensor, torch.Tensor, int, bool]:
        """
        Optimises the solutions using Model-Informed Variation. 

        Args:
            solutions: torch.Tensor
                The solutions to learn from. Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.
            fitnesses: torch.Tensor
                The list of fitnesses relating to each solution. Has shape N, where the ith fitness
                corresponds to the ith solution in the solutions tensor.
            change_tolerance: int
                Defines how many neutral or negative fitness changes can be made in a row before a 
                solution is deemed an optima during the optimisation process.
            encode: bool
                If true, the Encode method of varying will be used, and the Assign method otherwise.
                Default False.
            repair_solutions: bool
                If the problem has a repair method, that can be called after a change has been done to a solution
                to ensure that any changes still allows the solutions to be valid.
            deepest_only: bool
                If true, optimisation occurs at the deepest layer of the autoencoder only. If False, optimisation
                will occur at all levels of the autoencoder.
        
        Returns:
            A list containing the optimised solutions, their respective fitnesses, the number of
            evaluations used during the process, and a boolean that is true if one of the solutions
            is a global optima.
        """
        self.model.eval()
        evaluations = 0
        layers = [self.model.num_layers-1] if deepest_only else range(self.model.num_layers-1, 0, -1)
        for layer in layers:
            old_fitnesses = fitnesses.clone().detach()
            last_improve = torch.zeros_like(fitnesses, device=self.device)

            while True:
                new_solutions = self.model.vary(solutions, layer, encode)

                if repair_solutions:
                    new_solutions = self.problem.repair(new_solutions)
                evaluations += self.assess_changes(solutions, fitnesses, new_solutions,
                                                   change_tolerance, last_improve)
                if torch.any(fitnesses == self.problem.max_fitness): 
                    return (solutions, fitnesses, evaluations, True)
                if torch.all(last_improve > change_tolerance):
                    break   

        return (solutions, fitnesses, evaluations, False)
