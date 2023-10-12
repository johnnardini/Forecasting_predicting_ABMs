import numpy as np
from scipy.stats import qmc

def get_pulling_params():
    params = []
    
    Pm = 1.0
    Ppulls = np.round(np.linspace(0,1.0,11),2)

    for Ppull in Ppulls:
        params.append( (Pm, Ppull) )
        
    Ppull = 0.5
    Pms = np.round(np.linspace(0.1,1.0,10),2)
    #remove duplicate
    Pms = Pms[Pms!=1.0]
    for Pm in Pms:
        params.append( (Pm, Ppull) )
        
    return params

def get_adhesion_params():
    params = []
    
    Pm = 1.0
    Padhs = np.round(np.linspace(0,1.0,11),2)

    for Padh in Padhs:
        params.append( (Pm, Padh) )
        
    Padh = 0.5
    Pms = np.round(np.linspace(0.1,1.0,10),2)
    #remove duplicate
    Pms = Pms[Pms!=1.0]
    for Pm in Pms:
        params.append( (Pm, Padh) )
        
    return params

def get_adhesion_params_Padh_interpolation_Pm_fixed(data_type = "old"):
    
    params = []
    
    Pm = 1.0
    if data_type == "old":
        Padhs = np.linspace(0.5, 1.0, 6)
    elif data_type == "new":
        Padhs = np.linspace(0.55, 0.95, 5)

    for Padh in Padhs:
        params.append( (Pm, Padh) )
        
    return params

def get_adhesion_params_Pm_Padh_interpolation(data_type = "old"):
    
    params = []
    
    if data_type == "old":
        Pms = np.array([0.1, 0.5, 1.0])
        Padhs = np.linspace(0.5, 1.0, 6)
        for Pm in Pms:
            for Padh in Padhs:
                params.append( (Pm, Padh) )
    
    elif data_type == "new":
        sampler = qmc.LatinHypercube(d=2,centered=True,seed=1)
        Pm_Padh_sample = sampler.random(n=10)
        l_bounds = [0.1, 0.5]
        u_bounds = [1.0, 1.0]
        Pm_Padh_sample = np.round(qmc.scale(Pm_Padh_sample, l_bounds, u_bounds),4)
        for Pm, Padh in Pm_Padh_sample: 
            params.append( (Pm, Padh) )
    
        
    return params


def get_heterog_params():

    PmHBase = 0.25
    PmPBase = 1.0
    PadhBase = 0.33
    PpullBase = 0.33
    alphaBase = 0.5
    
    params = []
    
    ### Vary alpha
    for alpha in np.round(np.linspace(0.0,1.0,11),3):
        
        params.append( (PmHBase, PmPBase, PadhBase, PpullBase, alpha) ) 
        
    ### Vary Ppull
    for Ppull in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67]:
        params.append( (PmHBase, PmPBase, PadhBase, Ppull, alphaBase) ) 
        
    ### Vary Padh
    for Padh in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67]:
        
        params.append( (PmHBase, PmPBase, Padh, PpullBase, alphaBase) ) 
        
    ### Vary PmH
    for PmH in np.round(np.linspace(0.0,1.0,11),3):
        params.append( (PmH, PmPBase, PadhBase, PpullBase, alphaBase) ) 
        
    ### Vary PmP
    for PmP in np.round(np.linspace(0.5,1.5,11),3):
        params.append( (PmHBase, PmP, PadhBase, PpullBase, alphaBase) ) 
        
    return params

'''def get_heterog_LHC_params(data_type):
    
    if data_type == "Training":
        seed = 163
    elif data_type == "Testing":
        seed = 409
    
    sampler = qmc.LatinHypercube(d=5,centered=True,seed=seed)
    sample = sampler.random(n=20)
    #PmH, PmP, Padh, Ppull, alpha
    l_bounds = [0.0, 0.5,  0.0,  0.0, 0.0]
    u_bounds = [1.0, 1.5, 0.67, 0.67, 1.0]
    sample = np.round(qmc.scale(sample, l_bounds, u_bounds),4)
    
    params = []
    for s in sample:
        s_tmp = tuple(s)
        params.append(tuple(s_tmp))
    
    return params'''


def get_heterog_LHC_params(data_type):
    
    if data_type == "Training":
        seed = 5487
        n = 40
    elif data_type == "Testing":
        seed = 17
        n = 20
    
    sampler = qmc.LatinHypercube(d=3,centered=True,seed=seed)
    sample = sampler.random(n=n)
    #Padh, Ppull, alpha
    l_bounds = [0.0,  0.0, 0.0]
    u_bounds = [0.67, 0.67, 1.0]
    sample = np.round(qmc.scale(sample, l_bounds, u_bounds),3)
    
    params = []
    for s in sample:
        s_tmp = tuple((0.25, 1.0, s[0], s[1], s[2]))
        params.append(tuple(s_tmp))
    
    return params

    



