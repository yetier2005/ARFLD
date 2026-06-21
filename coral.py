import torch
import torch.nn as nn
from torch.autograd import Function, Variable

# CUDA = True if torch.cuda.is_available() else False


'''
MODELS
'''


def CORAL(source, target):
    d = source.data.shape[1]
    # d = 3

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss
