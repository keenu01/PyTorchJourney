"""
Basic Exercises on the 
fundamentlas of PyTorch
"""
#Creating a Tensor Using the Random Functon
x = t.rand(7,7) #The 7,7 is the shape of the Tensor meaning it is a 7x7 tensor it is in 1 dimension

#Matrix Multiplication
x1 = t.rand(6,5) # Creating a Random Tensor
x2 = t.rand(8,5) # Creating a Random Tensor

x2 = x2.T # T = Transpose. This reverses the Dimensions and is often used when doing Matrix Multiplication
print(t.mm(x1,x2)) # Matrix Multiplication is a common calculation in Deep Learning.
#Here is a good webstie to understand what MatrixMulti is: 
# https://www.mathsisfun.com/algebra/matrix-multiplying.html

#Setting Random Seeds
random_seed = t.manual_seed(10)#You can set 10 to any number you wish.

#Finding the MIN,MAX and Mean
x1 = t.rand(10,10) #Since we set the manual seed the result of rand will always be the same
#Find the Smallest number in tensor
print(x1.min())
#Finds the Biggest
print(x2.max())
#Calculates the mean
print(x2.mean())



