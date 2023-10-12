import sys, time
sys.path.append('../../')

from src.Modules.Utils.Imports import *
from src.Modules.Utils.ModelWrapper import ModelWrapper
from src.Modules.Models.BuildBINNs import BINN
import src.Modules.Loaders.DataFormatter as DF

from src.BINNs_training import BINN_training, plot_train_val_loss
from src.BINNs_model_selection import model_selection
from simulate_binn_PDEs import simulate_binn_DE
from get_params import get_adhesion_params

device = torch.device(GetLowestGPU(pick_from=[0,1,2,3]))

def to_torch(ndarray):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr

def to_numpy(x):
    return x.clone().detach().cpu().numpy()

if __name__ == '__main__':

    path = '../../data/'
    scenario = "simple_adhesion"
    Pp = 0.0
    
    BINN_model = 'DMLP'
    name = f"{BINN_model}"
    pde_weight = 1e4
    n = 25
    
    PmPpPadhs = np.array(get_adhesion_params())
    Pm, Pp, Padh = PmPpPadhs[int(sys.argv[1]),:]

    
    file_name = f'simple_adhesion_mean_{n}_Pm_{Pm}_Pp_{Pp}_Padh_{Padh}'
    
    #load data
    inputs, outputs, shape = DF.load_ABM_data(path+file_name+".npy", plot=False)
    x = np.unique(inputs[:,0])
    t = np.unique(inputs[:,1])
    tmax = np.max(t)
    
    t_upper_lim = .75*tmax

    t_training_index = (t <= t_upper_lim)
    t_train = t[t_training_index]
    
    for i in np.arange(5):
        #train on first 75% of timepoints
        training_index = inputs[:,1]<=t_upper_lim
        inputs_training = inputs[training_index,:]
        outputs_training = outputs[training_index,:]

        # initialize model
        binn = BINN(name,x,t_train,pde_weight=pde_weight)
        binn.to(device)

        t0 = time.time()
        binn,model = BINN_training(inputs_training, outputs_training, binn, f"{name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}",device)
        tf = time.time() - t0
        
        np.save(f"../../results/timing/ {name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}.npy",
               {'time':tf})

        plot_train_val_loss(model,f"training_{name}_training_replicate_{i}_{file_name}_pde_weight_{pde_weight}")
        
    model_selection((Pm, Pp, Padh), scenario = scenario, pde_weight=pde_weight, n=n)

    simulate_binn_DE((Pm, Pp, Padh),scenario)
