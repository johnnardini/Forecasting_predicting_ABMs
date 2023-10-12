import sys
sys.path.append('../../')

from src.Modules.Utils.Imports import *
from src.Modules.Utils.ModelWrapper import ModelWrapper
from src.Modules.Models.BuildBINNs import BINN
import src.Modules.Loaders.DataFormatter as DF

from src.BINNs_training import BINN_training
from src.BINNs_model_selection import model_selection
from simulate_binn_PDEs import simulate_binn_DE
from src.get_params import get_pulling_params, get_adhesion_params, get_adhesion_params_Padh_interpolation_Pm_fixed, get_adhesion_params_Pm_Padh_interpolation, get_heterog_params, get_heterog_LHC_params

#from src.custom_functions import to_torch, to_numpy

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))

if __name__ == '__main__':

    """
    This script trains BINN models to data from the Pulling ABM, Adhesion ABM, and Pulling & Adhesion ABM. Which dataset is trained on depends on the variable `model_name`.
    - When model_name = "simple_pulling", BINNs are trained to ABM simulations for forecasting the Pulling ABM
    - When model_name = "simple_adhesion", BINNs are trained to ABM simulations for forecasting the Adhesion ABM
    - When model_name = "adhesion_pulling", BINNs are trained to ABM simulations for forecasting the Pulling & Adhesion ABM
    - When model_name = "simple_adhesion_Padh_interp", BINNs are trained to ABM simulations for predicting the Adhesion ABM as Padh varies and Pm is fixed
    - When model_name = "simple_adhesion_Pm_Padh_interp", BINNs are trained to ABM simulations for predicting the Adhesion ABM as Pm and Padh vary
    - When model_name = "adhesion_pulling_LHC", BINNs are trained to ABM simulations for predicting the Pulling & Adhesion ABM as rmH and rmP are fixed and Padh, Ppull, and alpha are varied.
    
    Note:
    - The script saves timing information for model training, performs model selection over 5 trained BINN models. and simulates the final BINN-guided PDE from the selected BINN model.
    """
    
    scenario = "Pulling"
    
    path = '../../data/'
    
    #BINN_model = 'DMLP'
    name = "DMLP"
    pde_weight = 1e4
    n = 25
    
    model_name = "simple_pulling"
    #model_name = "simple_adhesion"
    #model_name = "adhesion_pulling"
    #model_name = "adhesion_pulling_LHC"
    #model_name = "simple_adhesion_Padh_interp"
    #model_name = "simple_adhesion_Pm_Padh_interp"

    if model_name == "simple_pulling":
        params = get_pulling_params()
        Pm, Ppull = params[int(sys.argv[1]),:]
        file_name = f'simple_pulling_mean_{n}_Pm_{Pm}_Ppull_{Ppull}'
        
    elif model_name == "simple_adhesion":
        params = get_adhesion_params()
        Pm, Padh = params[int(sys.argv[1]),:]
        file_name = f'simple_adhesion_mean_{n}_Pm_{Pm}_Padh_{Padh}'
        
    elif model_name == "adhesion_pulling":
        params = get_heterog_params()
        PmH, PmP, Padh, Ppull, alpha = np.round(params[int(sys.argv[1]),:],3)

        file_name = f'adhesion_pulling_mean_{n}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}'
        
    elif model_name == "adhesion_pulling_LHC":
        params = get_heterog_LHC_params("Training")
        file_name = f'adhesion_pulling_mean_{n}_PmH_{PmH}_PmP_{PmP}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}'

    elif model_name == "simple_adhesion_Pm_Padh_interp":
        seed_init = 500

        params = []
        params_tmp = get_adhesion_params_Pm_Padh_interpolation("old")
        ### Remove params already computed with "adhesion"
        for param in params_tmp:
            if param[0] != 1.0:
                params.append(param)
        Pm, Padh = params[int(sys.argv[1]),:]
        file_name = f'simple_adhesion_mean_{n}_Pm_{Pm}_Padh_{Padh}'
        
    #load data
    inputs, outputs, shape = DF.load_ABM_data(path+file_name+".npy", plot=False)
    x = np.unique(inputs[:,0])
    t = np.unique(inputs[:,1])
    tmax = np.max(t)
    
    #train on first 75% of timepoints
    t_upper_lim = .75*tmax
    t_training_index = (t <= t_upper_lim)
    t_train = t[t_training_index]
    
    training_index = inputs[:,1]<=t_upper_lim
    inputs_training = inputs[training_index,:]
    outputs_training = outputs[training_index,:]

    for i in np.arange(5):
        # initialize model
        binn = BINN(name,x,t_train,pde_weight=pde_weight)
        binn.to(device)
    
        t0 = time.time()
        binn,model = BINN_training(inputs_training, outputs_training, binn, f"{name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}",device)
        tf = time.time() - t0
        
        #record time to train model
        np.save(f"../../results/timing/ {name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}.npy",
               {'time':tf})

    #model selection for the selected BINN model
    model_selection(params, scenario = model_name, pde_weight=pde_weight, n=n)

    #Simualate the BINN-guided PDE for the selected BINN model    
    simulate_binn_DE(params,model_name)