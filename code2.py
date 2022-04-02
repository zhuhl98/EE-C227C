from tkinter import constants
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import cvxpy as cp
from tqdm import trange 

parser = argparse.ArgumentParser(description='Setting learning rate type')
# parser.add_argument("-l", "--learning_rate_type",\
#      help="Setting the learning rate type", default="c/sqrt(t)")
# parser.add_argument("-c", "--C",\
#      help="Constant for learning rate", type = float, default=0.5)
parser.add_argument("-l", "--L",\
      help="value of lambda", type = float, default=0.5)
parser.add_argument("-t", "--T",\
     help="Time steps", type = int, default=100)
args = parser.parse_args()
# print(args.learning_rate_type)
print(args.T)
print(args.L)
# print(args.C)

K, d = 5, 10

A = np.zeros((K,d,d), dtype = float)
b = np.zeros((K,d), dtype = float)

# print(A)
# construct A and b
for k in range(K):
    for i in range(d):
        for j in range(d):
            if (i < j):
                A[k,i,j] = math.exp((i+1)/(j+1)) \
                    * math.cos((i+1)*(j+1)) * math.sin(k+1)
            elif (i > j):
                A[k,i,j] = A[k,j,i]
        b[k,i] = math.exp((i+1)/(k+1)) * math.sin((i+1)*(k+1))
        A[k,i,i] = (i+1)/10* abs(math.sin(k+1)) + np.sum(np.abs(A[k,i]))

# print(A)
# print(b)

def f_k(x, k):
    return np.dot(x, np.dot(A[k], x)) - np.dot(b[k], x)

def f(x):
    return max([f_k(x, k) for k in range(K)])

T = args.T
lamb = args.L
# C = args.C

x1 = np.ones((10), dtype = float)
# print(f(x1))


x_iter = [x1]

def sub_gradient(x):
    k = np.argmax([f_k(x, k) for k in range(K)])
    g = 2*np.dot(A[k],x) - b[k]
    return g

def norm(x):
    return np.sqrt(np.sum(np.square(x)))

# with 1/sqrt(t) type learning rate

f_iter = [f(x1)]
f_best = [f(x1)]
f_plus = f(x1)

# a guess for optimal f which is computed in HW 1
f_star_guess = -0.8414004108659314

def solve_f_minus():
    x = cp.Variable(d)
    
    # print(max([f(xj) + cp.sum(cp.multiply(sub_gradient(xj),x-xj)) for xj in x_iter]))
    print(cp.max(cp.hstack(f(xj) + cp.sum(cp.multiply(sub_gradient(xj),x-xj)) for xj in x_iter)))
    obj = cp.Minimize(cp.max(cp.hstack(f(xj) + cp.sum(cp.multiply(sub_gradient(xj),x-xj)) for xj in x_iter)))
    prob = cp.Problem(obj)
    prob.solve()
    return prob.value

def proj(l):
    x_old = x_iter[-1]
    x_new = cp.Variable(d)
    obj = cp.Minimize(cp.sum(cp.square(x_old - x_new)))
    constraints = [cp.max(cp.hstack(f(xj) + cp.sum(cp.multiply(sub_gradient(xj),x-xj)) for xj in x_iter)) <= l]
    try:
        prob = cp.Problem(obj, constraints)
        prob.solve()
    except:
        return x_old
    return x_new.value

for t in trange(T-1, desc="level method", unit="sec"):
    x = x_iter[-1]

    f_minus = solve_f_minus()
    f_plus = min(f_plus, f(x))
    l = (1-lamb)*f_minus + lamb*f_plus
    x = proj(l)

    x_iter.append(x)
    f_iter.append(f(x))
    f_best.append(min(f_best[-1],f(x)))

f_opt = f_best[-1]
print(f_opt)

t_iter = np.arange(1,T+1, 1)

plt.loglog(t_iter, f_best - np.min([f_opt, f_star_guess]))
plt.xlabel('# of iteration')
plt.ylabel('sub-optimality gap')
plt.title('Convergence rate of level method for lambda = {}'.format(lamb))
plt.show()