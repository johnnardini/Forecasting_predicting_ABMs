import sys, importlib, time
sys.path.append('../../')

from src.Modules.Utils.Imports import *
from src.Modules.Models.BuildBINNs import BINN
from src.Modules.Utils.ModelWrapper import ModelWrapper

import src.Modules.Loaders.DataFormatter as DF

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))

from src.DE_simulation import fickian_diffusion, Reaction_Diffusion_eqn, simple_pulling_diffusion, simple_adhesion_diffusion, logistic_proliferation, no_proliferation
from scipy.integrate import odeint
import shutil

# helper functions
def to_torch(x):
    return torch.from_numpy(x).float().to(device)
def to_numpy(x):
    return x.detach().cpu().numpy()

def load_model(binn_name, x, t, save_name):

    # instantiate BINN
    binn = BINN(binn_name,x, t).to(device)

    # wrap model and load weights
    parameters = binn.parameters()
    model = ModelWrapper(
        model=binn,
        optimizer=None,
        loss=None,
        save_name=save_name)
    
    weight = '_best_val'
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

def unique_inputs(inputs):
    x = np.unique(inputs[:,0])
    t = np.unique(inputs[:,1])
    return x,t

def data_splitting(inputs,outputs,x,t,perc=0.75):
    
    tmax = np.max(t)

    #train on first 75% of timepoints
    training_index = inputs[:,1] <= perc*tmax
    testing_index  = inputs[:,1] >  perc*tmax
    
    inputs_training = inputs[training_index,:]
    outputs_training = outputs[training_index,:]
    inputs_testing = inputs[testing_index,:]
    outputs_testing = outputs[testing_index,:]

    t_train = np.unique(inputs_training[:,1])
    t_test = np.unique(inputs_testing[:,1])
    
    U_train = outputs_training.reshape((len(x),-1))
    U_test = outputs_testing.reshape((len(x),-1))
    
    return t_train, t_test, U_train, U_test

def DE_sim_train_test(x, t, Dq, Dr, IC, Diffusion_function, Growth_function):
    
    tmax = np.max(t)
    
    sol = odeint(Reaction_Diffusion_eqn, IC, t, args=(x, Dq, Dr, 
                                                      Diffusion_function, Growth_function))
    sol = sol.T
    sol_train = sol[:,t<=.75*tmax]
    sol_test  = sol[:,t >.75*tmax]
    
    return sol_train, sol_test

def model_selection(params, scenario, pde_weight, n, perc=0.75):
    
    path = '../../data/'
    
    if "simple_pulling" in scenario:
        filename_header = f"simple_pulling_mean_{n}"
        
        Pm, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Ppull_{Pinteraction}'    
 
    elif "simple_adhesion" in scenario:
        filename_header = f"simple_adhesion_mean_{n}"
        
        Pm, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Padh_{Pinteraction}' 
                
    elif "heterogeneous" in scenario:
        filename_header = f"adhesion_pulling_mean_{n}"
        
        PmH, PmP, Padh, Ppull, alpha = params
        
        file_name = f'{filename_header}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}' 
        
    ### BINN model information
    save_folder = "../../Weights/"
    weight = '_best_val'
    model_name = "DMLP"    

    ### load in data
    inputs, outputs, shape  = DF.load_ABM_data(path+file_name+".npy",plot=False)
    x = inputs[:,0]
    t = inputs[:,1]

    models = []
    binns = []

    for i in np.arange(5):
        binn_name  = f"{model_name}"
        save_name =  f"BINN_training_{binn_name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}"
        model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name,x=x,t=t)

        models.append(model)
        binns.append(binn)
        
    x,t = unique_inputs(inputs)

    t_train, t_test, U_train, U_test = data_splitting(inputs,
                                                      outputs,
                                                      x,
                                                      t,
                                                      perc=0.75)


    IC = U_train[:,0]

    MSE_binn_train = []
    count = 0
    
    for binn in binns:
        print(count)
        D_binn, G_binn = recover_binn_params(binn)
        sol_binn_train, sol_binn_test = DE_sim_train_test(x, 
                                                t, 
                                                [], 
                                                [],
                                                IC, 
                                                Diffusion_function = D_binn,
                                                Growth_function = G_binn)

        MSE_binn_train.append(np.linalg.norm(sol_binn_train - U_train))
        count+=1

    selected_replicate = np.argmin(MSE_binn_train)
    selected_file_name = f"{save_folder}BINN_training_{binn_name}_training_replicate_{selected_replicate}_{file_name}_pde_weight_{pde_weight}{weight}_model"
    save_to_filename = f"{save_folder}BINN_training_{binn_name}_{file_name}_pde_weight_{pde_weight}{weight}_model"

    shutil.copy(selected_file_name,
                save_to_filename)