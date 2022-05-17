import pandas as pd
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax


df_final = pd.read_csv('./dataset.csv')


b = 20000
n = 10

# Picking a small sample 
df_sample = df_final.sample(n=n)

# Viewing the sample 
df_sample

# Optimization Algorithm 
# Optimization Algorithm 
np.random.seed(1)

n = len(df_sample)
print('n',n)
x0 = np.random.randn(n)
c = df_sample['follower_count'].to_numpy()
print('c',c)
A = df_sample['CPP'].to_numpy()
print('A',A)

x = cp.Variable(n)
prob = cp.Problem(cp.Maximize(c.T@x),
                 [A.T @ x <= b,x >= 0,sum(x)<=4])

prob.solve()
# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)

# Making Observations on the output, Calculating views achieved and allocation
df_sample['Allocation'] = x.value
df_sample['Views'] = df_sample['Allocation'] * df_sample['follower_count']
df_sample['Cost'] = df_sample['Allocation'] * df_sample['CPP']
df_sample
