import torch as t
"""
These are the fundementals of PyTorch.
Tensors are the bulding blocks of Deep learning as it is the data that we input into
our neural networks.

Almost anything can be converted into a Tensor (some things harder than overs)
and over time You will understand the importance of Tensors and why we use them
in Deep Learning.
"""
#Basic Vector
t.tensor([[3,4]])

#Basic Matrix
t.tensor([[3,4],
          [5,4]])

#Basic Tensor
t.tensor([[[3,4,6],
           [4,5,7],
           [10,2,5]]])
#Random Tensor
t.rand(10,10)

#Ones
t.ones(10,10)

#zeros
t.zeros(10,10)

#A range
t.arange(start=0,end=100)
