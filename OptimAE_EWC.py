import torch
from torch.utils.data import DataLoader, TensorDataset
from OptimAE import OptimAEHandler

class EWCOptimAEHandler(OptimAEHandler):
    """
    Implement the Handler necessary to EWC
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
        self.model.ewc_terms={}
        
    
    def ewc_adapation(self, solutions: torch.Tensor) -> None:
        """
        Update the EWC terms (Fisher Information and Optimal Weight) used to compute the EWC loss.

        Parameters
        ----------
        solutions : torch.Tensor
            The solutions to use to compute the FIM.
            Has shape N x W, where N is the number of solutions
            in the population and W is the size of each solution.

        Returns
        -------
        None
            DESCRIPTION.

        """
        fisher = self.model.fim_computation(solutions)

        #replace
        if self.model.mode=="replace":
            self.model.ewc_terms=fisher
            
        #all
        elif self.model.mode=="all":
            if self.model.transited:
                for key in fisher.keys():
                    if key in self.model.ewc_terms.keys():
                        self.model.ewc_terms[key] += fisher[key]
                    else:
                        self.model.ewc_terms[key] = fisher[key]   
        
    def learn_from_population(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                              l1_coef: float = 0.0, batch_size: int = 16, epochs: int = 400, 
                              print_loss: bool = False) -> None:
        
        super().learn_from_population(solutions, optimizer,
                                  l1_coef, batch_size, epochs, 
                                  print_loss)
        """
        Method to make the AE learn from the population of solutions.
        Computes also EWC terms.
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
        self.ewc_adapation(solutions)
        if self.model.mode=="all":
            self.model.transited = False
    
    def learn_from_population_detail(self, solutions: torch.Tensor, optimizer: torch.optim.Optimizer,
                                     l1_coef: float = 0.0, batch_size: int = 16, epochs: int = 400,
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
        detailed_loss = super().learn_from_population_detail(solutions, optimizer,
                                                             l1_coef, batch_size, epochs, 
                                                             print_loss)
        
        self.ewc_adapation(solutions)
        if self.model.mode=="all":
            self.model.transited = False
        return detailed_loss