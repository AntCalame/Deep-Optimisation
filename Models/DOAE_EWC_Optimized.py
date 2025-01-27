import os
        
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
os.chdir(parent_directory)

import torch
import torch.nn.functional as F

from Models.DOAE2 import DOAE_detail

class DOAE_EWC(DOAE_detail):
    """
    Implement both All-EWC and Replace-EWC presented in the thesis:
    Reduced Forgetting Improves Deep Optimisation of Multiple Knapsack Problems
    By Antoine Calame
    """
    def __init__(self, input_size: int, dropout_prob: float, device: torch.device, param: float, mode: str, forgetting_factor: float=1.):
        """
        Constructor method for the AE. The encoder and decoder start of as empty models, as 
        layers get added to them during subsequent transitions.
        Modification needed for EWC are included.

        Args:
            input_size: int
                The size of the solutions in a given combinatorial optimisation problem.
            dropout_prob: float
                The amount of dropout that occurs in the input to the model.
            device: torch.device
                The device the model is loadeded onto.
            param: float
                The Fisher Multiplier used.
            mode: string
                Either "All" or "Replace"
        """
        super().__init__(input_size, dropout_prob, device)
        self.ewc_terms = {}
        self.ewc_coeff = param
        self.mode = mode
        self.forgetting_factor = forgetting_factor
        
    def fim_computation(self, x: torch.Tensor) -> dict:
        """
        Computation of the Fisher Information Matrix.

        Args:
            x : torch.Tensor
                The input to the model of size n x W, where n is the size of the batch and W is 
                the size of each solution.
        Returns
            dict
                Contains all Fisher Information associated with the parameter of the model.

        """
        fisher = {}
        polynom = {}
        
        self.eval()
        self.zero_grad()
        
        x.requires_grad = True
        
        size_pop = x.size(0)
        
        for i in range(size_pop):
            x_estimated_i = self.forward(x[i][None,...])[0]
            mse = F.mse_loss(x_estimated_i, x[i][None,...])
            mse.backward() 
            
            if i==0:
                for name, param in self.named_parameters():
                    key = name.split('.')[0] + ','.join(map(str, param.data.size()))
                    
                    fisher[key] = {"weight" : param.clone().detach(),
                                   "fim" : (param.grad.data ** 2)/size_pop}
                    
            elif i==size_pop-1:
                for name, param in self.named_parameters():
                    key = name.split('.')[0] + ','.join(map(str, param.data.size()))
                    
                    fisher[key]["fim"] += (param.grad.data ** 2)/size_pop
                    
                    polynom[key] = {"p0" : fisher[key]["fim"] * (fisher[key]["weight"]**2),
                                    "p1" : -2*fisher[key]["fim"] * fisher[key]["weight"],
                                    "p2" : fisher[key]["fim"]
                                    }
                    
            else:
                for name, param in self.named_parameters():
                    key = name.split('.')[0] + ','.join(map(str, param.data.size()))
                    
                    fisher[key]["fim"] += (param.grad.data ** 2)/size_pop
                    
            self.zero_grad()
            
        return polynom
    
    def loss_ewc(self) -> float:
        """
        Compute the term of the loss associated with EWC.

        Returns
            float
                EWC loss

        """
        
        loss = torch.tensor(0., requires_grad=True)
        
        for name,param in self.named_parameters():
            
            key = name.split('.')[0] + ','.join(map(str, param.data.size()))
            
            if key in self.ewc_terms.keys():
                
                loss = loss + (self.ewc_terms[key]["p0"]
                               +self.ewc_terms[key]["p1"]*param
                               +self.ewc_terms[key]["p2"]*(param**2)).sum()
                
        loss = loss * self.ewc_coeff/2 
        return loss
    
    def loss(self, x: torch.Tensor, recon: torch.Tensor, l1_coef: float) -> dict:
        """
        Compute all losses including: MSE, L1 regularization, L2 regularization, EWC loss 

        Parameters
        x : torch.Tensor
            The input to the model of size n x W, where n is the size of the batch and W is 
            the size of each solution.
        recon : torch.Tensor
            The reconstruction of x that the model outputs.
        l1_coef : float
            The coefficient of the L1 loss term.

        Returns
        dict
            The loss dictionary containing the total loss, 
            reconstruction error and L1 loss, L2 loss and EWC loss.

        """
        dl = super().loss(x, recon, l1_coef)
        
        ewc_loss = self.loss_ewc()
        dl["ewc"] = ewc_loss
        dl["loss"] += ewc_loss

        return dl

    def transition(self, hidden_size : int) -> None:
        """
        Adds a new layer to the model. This should be called after the solutions have been 
        optimised with Model-Informed Variation.
        It integrates a value used in the All-EWC.
        If self.transited=True, then terms regarding the new layers will be added to the loss.

        Args:
            hidden_size: int
                The size of the next hidden layer to be added.
        """
        super().transition(hidden_size)
        if self.mode=="all":
            self.transited = True

