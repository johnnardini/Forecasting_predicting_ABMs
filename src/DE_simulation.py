import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy import integrate
import matplotlib as mpl
from scipy import interpolate
import time

from scipy import sparse
from scipy.integrate import odeint
from inspect import signature


def fickian_diffusion(u,q):
    return q[0]*np.ones(u.shape)

def simple_pulling_diffusion(u,q):
    return q[0]*(1.0+3*q[1]*u**2)

def simple_adhesion_diffusion(u,q):
    return q[0]*(1+q[1]*(3*u**2 - 4*u))

def logistic_proliferation(u,q):
    return q[0]*(1.0-u)

def no_proliferation(u,q):
    return np.zeros(u.shape)

def D_HH(H, P, T, q):
    #q = [DH, DP, Pa, Pp]
    return q[0]*(1-T - 4*q[2]*(1-T)*H) + q[1]*q[3]*((1-T)*P)

def D_HP(H, P, T, q):
    #q = [DH, DP, Pa, Pp]
    return q[1]*( -q[3]*((1-T)*H) )

def D_HT(H, P, T, q):
    #q = [DH, DP, Pa, Pp]
    return q[0]*(H - q[2]*H**2) + q[1]*( 3*q[3]*H*P )

def D_PH(H, P, T, q):
    #q = [DH, DP, Pa, Pp]
    return q[1]*( -3*q[2]*P*(1-T) )

def D_PP(H, P, T, q):
    #q = [DH, DP, Pa, Pp]
    return q[1]*( 1-T - q[2]*H*(1-T) )

def D_PT(H, P, T, q):
    #q = [DH, DP, Pa, Pp]
    return q[1]*( P - q[2]*H*P + 3*q[3]*P**2 )


def Diffusion_eqn(u,t,x,D,diffusion_function):
    
    dx = x[1] - x[0]
    
    #check number of function inputs
    sig = signature(diffusion_function)
    if len(sig.parameters) == 1:
        D_matrix = D_u(diffusion_function(u),dx)
    elif len(sig.parameters) == 2:
        D_matrix = D_u(diffusion_function(u,D),dx)
    
    return D_matrix.dot(u)

def Heterogeneous_Diffusion_eqn(u,t,x,q):
    
    dx = x[1] - x[0]
    n = len(u)//2
    
    H,P = u[:n], u[n:]
    T   = H + P
    
    DHH_matrix = D_u( D_HH(H, P, T, q) , dx)
    DHP_matrix = D_u( D_HP(H, P, T, q) , dx)
    DHT_matrix = D_u( D_HT(H, P, T, q) , dx)
    
    DPH_matrix = D_u( D_PH(H, P, T, q) , dx)
    DPP_matrix = D_u( D_PP(H, P, T, q) , dx)
    DPT_matrix = D_u( D_PT(H, P, T, q) , dx)
    
    dHdt = DHH_matrix.dot(H) + DHP_matrix.dot(P) + DHT_matrix.dot(T)
    dPdt = DPH_matrix.dot(H) + DPP_matrix.dot(P) + DPT_matrix.dot(T)
    
    return np.hstack( (dHdt, dPdt) )

def Reaction_Diffusion_eqn(u,t,x,D,r,diffusion_function,growth_function):
    
    dx = x[1] - x[0]
    
    #check number of function inputs
    sigD = signature(diffusion_function)
    if len(sigD.parameters) == 1:
        D_matrix = D_u(diffusion_function(u),dx)
    elif len(sigD.parameters) == 2:
        D_matrix = D_u(diffusion_function(u,D),dx)
        
    sigG = signature(growth_function)
    if len(sigG.parameters) == 1:
        growth_rate = growth_function(u)
    elif len(sigG.parameters) == 2:
        growth_rate = growth_function(u,r)
    
    return D_matrix.dot(u) + growth_rate*u




def D_u(D,dx):
    
    '''
    Create the Matrix operator for (D(u)u_x)_x, where D is a vector of values of D(u),
    and dx is the spatial resolution based on methods from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    '''
    n = len(D)
    
    D_ind = np.arange(n)

    #first consruct interior portion of D
    #exclude first and last point and include those in boundary
    D_ind = D_ind[1:-1] 
    #Du_mat : du_j/dt = [(D_j + D_{j+1})u_{j+1}
    #                   -(D_{j-1} + 2D_j + D_{j+1})u_j
    #                   + (D_j + D_{j-1})u_{j-1}] 
    Du_mat_row = np.hstack((D_ind,D_ind,D_ind))
    Du_mat_col = np.hstack((D_ind+1,D_ind,D_ind-1))
    Du_mat_entry = (1.0/(2*dx**2))*np.hstack((D[D_ind+1]+D[D_ind],
                   -(D[D_ind-1]+2*D[D_ind]+D[D_ind+1]),D[D_ind-1]+D[D_ind]))
    
    #boundary points
    Du_mat_row_bd = np.array((0,0,n-1,n-1))
    Du_mat_col_bd = np.array((0,1,n-1,n-2))
    Du_mat_entry_bd = (1.0/(2*dx**2))*np.array((-2*(D[0]+D[1]),
                    2*(D[0]+D[1]),-2*(D[-2]+D[-1]),2*(D[-2]+D[-1])))
    #add in boundary points
    Du_mat_row = np.hstack((Du_mat_row,Du_mat_row_bd))
    Du_mat_col = np.hstack((Du_mat_col,Du_mat_col_bd))
    Du_mat_entry = np.hstack((Du_mat_entry,Du_mat_entry_bd))

    return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)))

def DE_sim(x, t, q, IC, Diffusion_function):
    
    sol = odeint(Diffusion_eqn, IC, t, args=(x, q, Diffusion_function))
    sol = sol.T
    return sol
