import numpy as np
from scipy.optimize import *



def f(z):
    a = z[0]
    b= z[1]
    f_ = np.empty((2))
    f_[0]= a**2+b-21
    f_[1]= a**2+t*b-1

    return f_
global t
t=3


z=fsolve(func=f,x0=np.array([1,0]))
print(z)