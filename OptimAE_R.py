import torch
from torch.utils.data import DataLoader, TensorDataset

from COProblems.OptimisationProblem import OptimisationProblem
from Models.DOAE import DOAE
from OptimAE import OptimAEHandler

class ROptimAEHandler(OptimAEHandler):
    """
    Implement the Handler necessary to Rehearsal-DO
    """
    def __init__(self, model, problem, device):
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
        self.memory = None
        
        
    def update_memory(self, pop: torch.Tensor):
        """
        Function used to construct and update the memory

        Parameters
            pop : torch.Tensor
                Simply the population used to update the memory.
                Has shape N x W, where N is the number of solutions
                in the population and W is the size of each solution.

        """
        if self.memory is None:
            self.memory = pop.clone().detach()
        else:
            self.memory = torch.cat([self.memory,pop],0)
            
    
    def learn_from_population(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              l1_coef: float = 0.0, batch_size: int = 16, 
                              p: float = 0.0, epochs: int = 400, 
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
        
        batch_size_pop = round(batch_size*(1-p))
        batch_size_memory = batch_size - batch_size_pop
        
        dataset = DataLoader(TensorDataset(solutions), batch_size=batch_size_pop, shuffle=True)
        dataset_memory = DataLoader(TensorDataset(self.memory), batch_size=batch_size_memory, shuffle=True)
        
        for epoch in range(epochs):
            for i,(x,y) in enumerate(zip(dataset, dataset_memory)):
                loss = self.model.learn_from_sample(torch.cat([x[0],y[0]]), optimizer, l1_coef)
                total_recon += loss["recon"]
            if print_loss:
                if (epoch+1) % 10 == 0:
                    print("Epoch {}/{} - Recon Loss = {}".format(
                        epoch+1,epochs,total_recon/(10*len(dataset))
                    ))
                    total_recon = 0
                    
    def learn_from_population_detail(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                                     l1_coef: float = 0.0, batch_size: int = 16, 
                                     p: float = 0.0, epochs: int = 400,
                                     print_loss: bool = False) -> None:
        """
        Method to make the AE learn from the population of solutions.
        Output the detailed loss dict.

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
                
        Results:
            dict
                Contains the loss detailed per epoch 
                Including Total, MSE, L1 regularization, L2 regularization and EWC.
        """
        total_recon = 0
        detailed_loss = []
        
        batch_size_pop = round(batch_size*(1-p))
        batch_size_memory = batch_size - batch_size_pop
        
        dataset = DataLoader(TensorDataset(solutions), batch_size=batch_size_pop, shuffle=True)
        dataset_memory = DataLoader(TensorDataset(self.memory), batch_size=batch_size_memory, shuffle=True)
        
        for epoch in range(epochs):
            for i,(x,y) in enumerate(zip(dataset, dataset_memory)):
                loss = self.model.learn_from_sample(torch.cat([x[0],y[0]]), optimizer, l1_coef)
                total_recon += loss["recon"]
                detailed_loss += [loss]
                
                for i,j in loss.items():
                    print(j.size())
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