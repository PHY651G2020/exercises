
# coding: utf-8

# ## 1. MLE

# In[ ]:


import numpy as np

# 1. MLE
def lnL(xx,yy,a,b):
    """function returning -2*ln(L)"""
    ... #TODO

def sl_MLE(xx,yy):
    """compute the MLE for the straight line (sl) parameters"""
    a = ... #TODO
    b = ... #TODO
    return (a,b)

def sl_MLE_var(xx,yy):
    """compute the covariance matrix of the MLE"""
    V11 = ... #TODO
    V12 = ... #TODO
    V21 = ... #TODO
    V22 = ... #TODO
    return ((V11,V12),(V21,V22))

# position of the detector hits
xx12 = np.array([10,11,12,20,21,22])
yy12 = np.array([4.0,3.8,3.6,5.2,4.9,4.8])

# sub-arrays with only station 1 or only station 2
xx1 = xx12[[0,1,2]]
xx2 = xx12[[3,4,5]]
yy1 = yy12[[0,1,2]]
yy2 = yy12[[3,4,5]]

print ("first station only: V(MLE) = ", sl_MLE_var(xx1,yy1))
print ("second station only: V(MLE) = ", sl_MLE_var(xx2,yy2))
print ("both stations: V(MLE) = ", sl_MLE_var(xx12,yy12))
mle12 = sl_MLE(xx12,yy12)
print ("MLE for both stations: ", mle12)

mle1 = sl_MLE(xx1,yy1)
mle2 = sl_MLE(xx2,yy2)


# ## 2. draw $-2 \ln \mathcal{L}$

# In[ ]:


from scipy import optimize
import matplotlib.pyplot as plt


# In[ ]:


# 2. draw -2 ln L

# 'wrappers' for the optimization (which happens with respect to the first parameter)
def lnL_args_ab(x,*args):
    """for minimizing with respect to both a and b"""
    a = x[0]
    b = x[1]
    xx = args[0]
    yy = args[1]
    return lnL(xx,yy,a,b)

def lnL_args_a(x,*args):
    """for minimizing with respect to only b"""
    a = args[2]
    b = x
    xx = args[0]
    yy = args[1]
    return lnL(xx,yy,a,b)

# -2ln(L) values for both stations combined
lh_vals12 = []
# profile -2ln(L) values for both stations combined
proflh_vals12 = []
# -2ln(L) values for the first station only
lh_vals1 = []
# profile -2ln(L) values for the first station only
proflh_vals1 = []
# -2ln(L) values for the second station only
lh_vals2 = []
# profile -2ln(L) values for the second station only
proflh_vals2 = []

# minimum -2ln(L) for both stations combined
minlh12 = ... #TODO
# minimum -2ln(L) for the first station only
minlh1 = ... #TODO
# minimum -2ln(L) for the second station only
minlh2 = ... #TODO

# array with the a values that we are going to scan
a_vals = np.linspace(0,15,150)

for ai in a_vals:
    # 12
    lh_vals12.append(...) #TODO
    # LH minimisation: use https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
    bhathat = ...#TODO
    proflh_vals12.append(...) #TODO
    # 1
    lh_vals1.append(...) #TODO
    bhathat = ... #TODO
    proflh_vals1.append(...) #TODO
    # 2
    lh_vals2.append(...) #TODO
    bhathat = optimize.fmin(...) #TODO
    proflh_vals2.append(...) #TODO

# plotting
fix, ax = plt.subplots()
ax.plot(a_vals, proflh_vals1, label='$L_{prof,1} (a)$', linestyle='-', color='r')
ax.plot(a_vals, lh_vals1, label='$L_{1} (a,\hat{b})$', linestyle='--', color='r')
ax.plot(a_vals, proflh_vals2, label='$L_{prof,2} (a)$', linestyle='-', color='b')
ax.plot(a_vals, lh_vals2, label='$L_{2} (a,\hat{b})$', linestyle='--', color='b')
ax.plot(a_vals, proflh_vals12, label='$L_{prof,1+2} (a)$', linestyle='-', color='black')
ax.plot(a_vals, lh_vals12, label='$L_{1+2} (a,\hat{b})$', linestyle='--', color='black')
ax.set_ybound(0,6)
ax.set_xlabel('a')
ax.set_ylabel('$-2 \Delta \ln(\mathcal{L})$')
ax.legend()
plt.show()


# ## 3. Add information
# $$\langle b \rangle = 0.1,\quad \sigma_b = 0.05$$

# In[ ]:


def lnL(xx,yy,a,b):
    """modified -2*ln(L) which accounts for the additional information"""
    ... #TODO

def sl_MLE(xx,yy):
    """return the MLE (estimated numerically)"""
    # LH minimisation: use https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
    mle = ... #TODO
    return mle

mle12 = sl_MLE(xx12,yy12)
mle1 = sl_MLE(xx1,yy1)
mle2 = sl_MLE(xx2,yy2)

# -2ln(L) values for both stations combined
lh_vals12 = []
# profile -2ln(L) values for both stations combined
proflh_vals12 = []
# -2ln(L) values for the first station only
lh_vals1 = []
# profile -2ln(L) values for the first station only
proflh_vals1 = []
# -2ln(L) values for the second station only
lh_vals2 = []
# profile -2ln(L) values for the second station only
proflh_vals2 = []

# minimum -2ln(L) for both stations combined
minlh12 = ... #TODO
# minimum -2ln(L) for the first station only
minlh1 = ... #TODO
# minimum -2ln(L) for the second station only
minlh2 = ... #TODO

# array with the a values that we are going to scan
a_vals = np.linspace(0,15,150)

for ai in a_vals:
    # 12
    lh_vals12.append(...) #TODO
    # LH minimisation: use https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
    bhathat = ...#TODO
    proflh_vals12.append(...) #TODO
    # 1
    lh_vals1.append(...) #TODO
    bhathat = ... #TODO
    proflh_vals1.append(...) #TODO
    # 2
    lh_vals2.append(...) #TODO
    bhathat = optimize.fmin(...) #TODO
    proflh_vals2.append(...) #TODO

# plotting
fix, ax = plt.subplots()
ax.plot(a_vals, proflh_vals1, label='$L_{prof,1} (a)$', linestyle='-', color='r')
ax.plot(a_vals, lh_vals1, label='$L_{1} (a,\hat{b})$', linestyle='--', color='r')
ax.plot(a_vals, proflh_vals2, label='$L_{prof,2} (a)$', linestyle='-', color='b')
ax.plot(a_vals, lh_vals2, label='$L_{2} (a,\hat{b})$', linestyle='--', color='b')
ax.plot(a_vals, proflh_vals12, label='$L_{prof,1+2} (a)$', linestyle='-', color='black')
ax.plot(a_vals, lh_vals12, label='$L_{1+2} (a,\hat{b})$', linestyle='--', color='black')
ax.set_ybound(0,6)
ax.set_xlabel('a')
ax.set_ylabel('$-2 \Delta \ln(\mathcal{L})$')
ax.legend()
plt.show()

