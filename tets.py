import os
os.system('clear')
from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import pickle


with open('Solution1.pickle', 'rb') as f:
    sol1 = pickle.load(f)


with open('Solution.pickle', 'rb') as f:
    sol = pickle.load(f)


sol1_x1 = sol1['x1']
sol1_u1 = sol1['u1']
sol_x1 = sol['x1']
sol_u1 = sol['u1']

sol1_u1 - sol_u1




