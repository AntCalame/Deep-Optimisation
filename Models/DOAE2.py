import torch
import torch.nn.functional as F
from torch import nn

from Models.DOAE import DOAE

class DOAE_detail(DOAE):
    """
    Same algorithm as DOAE in DOAE.py
    The only difference is the ability to capture the detailled loss during the optimisation process.
    """
    def __init__(self, input_size: int, dropout_prob: float, device: torch.device):
        """
        Constructor method for the AE. The encoder and decoder start of as empty models, as 
        layers get added to them during subsequent transitions.

        Args:
            input_size: int
                The size of the solutions in a given combinatorial optimisation problem.
            dropout_prob: float
                The amount of dropout that occurs in the input to the model.
            device: torch.device
                The device the model is loadeded onto.
        """
        super().__init__(input_size, dropout_prob, device)
        
    def loss(self, x: torch.Tensor, recon: torch.Tensor, l1_coef: float) -> dict:
        """
        Calculates the loss function. This is done by adding the MSE of the input and the 
        reconstruction given by the AE, as well as an L1 term multiplied by a coefficient.
        L2 is not included in this loss function as that is handled by the optimizer in 
        Pytorch.

        Args:
            x: torch.Tensor
                The input to the model of size n x W, where n is the size of the batch and W is 
                the size of each solution.
            recon: torch.Tensor
                The reconstruction of x that the model outputs.
            l1_coef: float
                The coefficient of the L1 loss term.

        Returns:
            The loss dictionary containing the total loss, reconstruction error and L1 loss.
        """
        mse = F.mse_loss(x, recon)
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum((p**2).sum() for p in self.parameters())
        loss = mse + l1_coef * l1_loss
        return {"loss" : loss, "recon" : mse, "l1" : l1_loss, "l2" : l2_loss}