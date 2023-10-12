import sys, importlib, time
sys.path.append('../../')
#sys.path.append('../../src/')

from src.Modules.Utils.Imports import *
from src.Modules.Models.BuildBINNs import BINN
from src.Modules.Utils.ModelWrapper import ModelWrapper

import src.Modules.Utils.PDESolver as PDESolver
import src.Modules.Loaders.DataFormatter as DF

from src.DE_simulation import fickian_diffusion, Diffusion_eqn, simple_pulling_diffusion, simple_adhesion_diffusion, Heterogeneous_Diffusion_eqn
from scipy.integrate import odeint
device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))

def load_model(binn_name, x, t, save_name):

    weight = '_best_val'
    # instantiate BINN
    binn = BINN(binn_name,x, t).to(device)

    # wrap model and load weights
    parameters = binn.parameters()
    model = ModelWrapper(
        model=binn,
        optimizer=None,
        loss=None,
        save_name=save_name)
    
    #model.save_name += '_' + binn.name + weight
    model.save_name += weight
    model.load(model.save_name + '_model', device=device)

    return model, binn


def recover_binn_params(binn):
    # learned diffusion term
    def D(u):
        D = binn.diffusion(to_torch(u)[:, None])
        return to_numpy(D).reshape(-1)

    # learned growth term
    def G(u):
        r = binn.growth(to_torch(u)[:, None])
        return to_numpy(r).reshape(-1)  

    return D, G

# helper functions
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()

def unique_inputs(inputs):
    x = np.unique(inputs[:,0])
    t = np.unique(inputs[:,1])
    return x,t

def DE_sim(x, t, q, IC, Diffusion_function, PDE_type = "1d"):
    
    if PDE_type == "1d":
        sol = odeint(Diffusion_eqn, IC, t, args=(x, q, Diffusion_function))
    elif PDE_type == "heterogeneous":
        sol = odeint(Heterogeneous_Diffusion_eqn, IC, t, args=(x, q))
        sol = sol[:, :len(x)] + sol[:, len(x):]
    sol = sol.T
    
    return sol

def simulate_binn_DE(params,scenario):

    print(params)
    
    ### BINN model information
    path = '../../data/'
    save_folder = "../../Weights/"
    model_name = 'DMLP'
    weight = '_best_val'
    surface_weight = 1.0
    pde_weight = 1e4
    
    if "simple_pulling" in scenario:
        
        Pm, Pinteraction = np.round(params,3)
        filename_header = f"{scenario}_mean_25"
        file_name = f'{filename_header}_Pm_{Pm}_Ppull_{Pinteraction}'
        
    elif "simple_adhesion" in scenario:
        
        Pm, Pinteraction = np.round(params,3)
        filename_header = f"{scenario}_mean_25"
        file_name = f'{filename_header}_Pm_{Pm}_Padh_{Pinteraction}'    
        
    elif "adhesion_pulling" in scenario:
        
        PmH, PmP, Padh, Ppull, alpha = np.round(params,3)
        
        filename_header = f"{scenario}_mean_25"
        file_name = f'{filename_header}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}'    

    ### load in data
    inputs, outputs, shape  = DF.load_ABM_data(path+file_name+".npy",plot=False)
    x,t = unique_inputs(inputs)
    U = outputs.reshape((len(x),-1))
    
    ### load in binn
    binn_name  = f"{model_name}"
    save_name =  f"BINN_training_{binn_name}_{file_name}_pde_weight_{pde_weight}"
    model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name,x=x,t=t)

    ### Simulate PDE
    IC = U[:,0]

    D_binn, G_binn = recover_binn_params(binn)

    t0 = time.time()
    sol_binn = DE_sim(x, 
                      t, 
                      [], 
                      IC, 
                      Diffusion_function = D_binn)
    tf = time.time() - t0

    PDE_file_name = f"../../results/PDE_sims/PDE_sim_{binn_name}_{file_name}_pde_weight_{pde_weight}.npy"

    data = {}
    data['x'] = x
    data['t'] = t
    data['U_sim'] = sol_binn
    data['U_data'] = U
    data['time'] = tf
    
    if "simple" in scenario:
        data['Pm'] = Pm
        data['Pint'] = Pinteraction
    elif "adhesion_pulling" in scenario:
        PmH, PmP, Pp, Padh, Ppull, alpha
        data['PmH'] = PmH
        data['PmP'] = PmP        
        data['Padh'] = Padh        
        data['Ppull'] = Ppull
        data['alpha'] = alpha        

    np.save(PDE_file_name,data)
    
    return sol_binn
    
def simulate_binn_DE_one_input(inputs):
    assert type(inputs[0]) == tuple
    assert type(inputs[1]) == str

    sol = simulate_binn_DE(inputs[0],inputs[1])   
    
    return sol
    
if __name__ == '__main__':
    
    inputs = []

    '''Pints = np.round(np.linspace(0.0,1.0,11),2)
    Pms = np.round(1.0*np.ones((11,)),2)
    Pps = np.round(np.zeros((11,)),2)

    for Pm, Pp, Pint in zip(Pms, Pps, Pints):
        inputs.append( [(Pm, Pp, Pint),
                        "pulling"] )

    Pints = np.round(.5*np.ones((10,)),2)
    Pms = np.round(np.linspace(0.1,1.0,10),2)
    for Pm, Pp, Pint in zip(Pms, Pps, Pints):
        inputs.append( [(Pm, Pp, Pint),
                        "pulling"] )

    Pints = np.round(np.linspace(0.0,1.0,11),2)
    Pms = np.round(1.0*np.ones((11,)),2)
    
    for Pm, Pp, Pint in zip(Pms, Pps, Pints):
        inputs.append( [(Pm, Pp, Pint),
                        "adhesion"] )

    Pints = np.round(.5*np.ones((10,)),2)
    Pms = np.round(np.linspace(0.1,1.0,10),2)
    for Pm, Pp, Pint in zip(Pms, Pps, Pints):
        inputs.append( [(Pm, Pp, Pint),
                        "adhesion"] )'''

    '''###### Heterogeneous data
    ### Vary alpha
    PmH = 0.25
    PmP = 1.0
    Pp = 0.0
    Padh = 0.33
    Ppull = 0.33
    for alpha in np.linspace(0.0,1.0,11):
        
        tmp_array = (PmH, PmP, Pp, Padh, Ppull, alpha)
        inputs.append( [tmp_array,
                        "adhesion_pulling"] )
    
    ### Vary Ppull
    PmH = 0.25
    PmP = 1.0
    Pp = 0.0
    alpha = 0.5
    Padh = 0.33
    for Ppull in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67]:
        
        tmp_array = (PmH, PmP, Pp, Padh, Ppull, alpha)
        inputs.append( [tmp_array,
                        "adhesion_pulling"] )
        
    ### Vary Padh
    PmH = 0.25
    PmP = 1.0
    Pp = 0.0
    alpha = 0.5
    Ppull = 0.33
    for Padh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67]:
        tmp_array = (PmH, PmP, Pp, Padh, Ppull, alpha)
        inputs.append( [tmp_array,
                        "adhesion_pulling"] )'''
        
    ### Vary PmH
    PmP = 1.0
    Pp = 0.0
    alpha = 0.5
    Ppull = 0.33
    Padh = 0.33
    for PmH in np.linspace(0.1,1.0,10):
        tmp_array = (PmH, PmP, Pp, Padh, Ppull, alpha)
        inputs.append( [tmp_array,
                        "adhesion_pulling"] )
    
    ### Vary PmP
    PmH = 0.25
    Pp = 0.0
    alpha = 0.5
    Ppull = 0.33
    Padh = 0.33
    for PmP in np.linspace(0.5,1.5,11):
        tmp_array = (PmH, PmP, Pp, Padh, Ppull, alpha)
        inputs.append( [tmp_array,
                        "adhesion_pulling"] )
    
    #pool = mp.Pool(mp.cpu_count())
    #results = pool.map(simulate_binn_DE_one_input, inputs)
    #pool.close()
    for input_val in inputs:
        simulate_binn_DE_one_input(input_val)

    
