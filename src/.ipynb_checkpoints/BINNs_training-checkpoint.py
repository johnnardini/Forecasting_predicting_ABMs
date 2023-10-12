import sys
sys.path.append('../')

from src.Modules.Utils.Imports import *
from src.Modules.Utils.ModelWrapper import ModelWrapper
from src.Modules.Models.BuildBINNs import BINN
import src.Modules.Loaders.DataFormatter as DF


def to_torch(ndarray,device):
    arr = torch.tensor(ndarray, dtype=torch.float)
    arr.requires_grad_(True)
    arr = arr.to(device)
    return arr
def to_numpy(x):
    return x.clone().detach().cpu().numpy()

def BINN_training(inputs, outputs, binn, binn_save_name, device):
    
    N = len(inputs)
    split = int(0.8*N)
    p = np.random.permutation(N)
    x_train = to_torch(inputs[p[:split]],device)
    y_train = to_torch(outputs[p[:split]],device)
    x_val = to_torch(inputs[p[split:]],device)
    y_val = to_torch(outputs[p[split:]],device)
    
    # compile 
    parameters = binn.parameters()
    opt = torch.optim.Adam(parameters, lr=1e-3)
    model_only_SF_loss = ModelWrapper(
        model=binn,
        optimizer=opt,
        loss=binn.gls_loss,
        augmentation=None,
        save_name=f"../../Weights/SF_only_{binn_save_name}")

    epochs = int(1e4)
    batch_size = 1000
    rel_save_thresh = 0.01

    # first only train the surface fitter
    model_only_SF_loss.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=None,
        verbose=0,
        validation_data=[x_val, y_val],
        early_stopping=1e3,
        rel_save_thresh=rel_save_thresh)

    # Now create a second model that will also train the surface fitter and PDE loss
    model = ModelWrapper(
        model=binn,
        optimizer=opt,
        loss=binn.loss,
        augmentation=None,
        save_name=f"../../Weights/BINN_training_{binn_save_name}")

    epochs = int(1e6)
    batch_size = 1000
    rel_save_thresh = 0.01

    # train jointly
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=None,
        verbose=0,
        validation_data=[x_val, y_val],
        early_stopping=1e5,#5e4
        rel_save_thresh=rel_save_thresh)
    
    return binn,model
    
def plot_train_val_loss(model,dataName):
    
    # load training errors
    total_train_losses = model.train_loss_list
    total_val_losses = model.val_loss_list

    
    rel_save_thresh = 0.01
    # find where errors decreased
    train_idx, train_loss, val_idx, val_loss = [], [], [], []
    best_train, best_val = 1e12, 1e12
    for i in range(len(total_train_losses)-1):
        rel_diff = (best_train - total_train_losses[i])
        rel_diff /= best_train
        if rel_diff > rel_save_thresh:
            best_train = total_train_losses[i]
            train_idx.append(i)
            train_loss.append(best_train)
        rel_diff = (best_val - total_val_losses[i])
        rel_diff /= best_val
        if rel_diff > rel_save_thresh:
            best_val = total_val_losses[i]
            val_idx.append(i)
            val_loss.append(best_val)
    idx = np.argmin(val_loss)

    # plot
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 2, 1)
    plt.semilogy(total_train_losses, 'b')
    plt.semilogy(total_val_losses, 'r')
    plt.semilogy(val_idx[idx], val_loss[idx], 'ko')
    plt.legend(['train mse', 'val mse', 'best val'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title('Train/Val errors')
    plt.grid()
    ax = fig.add_subplot(1, 2, 2)
    plt.semilogy(train_idx, train_loss, 'b.-')
    plt.semilogy(val_idx, val_loss, 'r.-')
    plt.legend(['train mse', 'val mse'])
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.title(f'Train/Val improvements \n {dataName}')
    plt.grid()
    
    plt.savefig(f"../../results/figures/trainval_{dataName}.pdf",format="pdf") 
    
def BINN_model_selection(params, scenario):
    
    path = '../../data/'

    if scenario == "simple_pulling":
        diffusion = simple_pulling_diffusion  
        growth = no_proliferation
        filename_header = "simple_pulling_mean_5"
        model_name = 'DMLP'
        surface_weight = 1.0
        pde_weight = 10.0
        
        Pm, Pp, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Pp_{Pp}_Ppull_{Pinteraction}'    
 
    elif scenario == "simple_adhesion":
        diffusion = simple_adhesion_diffusion  
        growth = no_proliferation
        filename_header = "simple_adhesion_mean_5"
        interaction_param = "Padh"
        model_name = 'DMLP'
        surface_weight = 1.0
        pde_weight = 10.0
        
        Pm, Pp, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Pp_{Pp}_Padh_{Pinteraction}' 
        
    elif scenario == "proliferation_pulling":
        diffusion = simple_pulling_diffusion
        growth = logistic_proliferation
        model_name = 'DMLP_GMLP'
        surface_weight = 1.0
        pde_weight = 1.0
        
        Pm, Pp, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Pp_{Pp}_Ppull_{Pinteraction}' 
        
    elif scenario == "proliferation_adhesion":
        diffusion = simple_adhesion_diffusion
        growth = logistic_proliferation
        model_name = 'DMLP_GMLP'
        surface_weight = 1.0
        pde_weight = 1.0
        
        Pm, Pp, Pinteraction = params
        file_name = f'{filename_header}_Pm_{Pm}_Pp_{Pp}_Padh_{Pinteraction}' 
    


    ### BINN model information
    save_folder = "../../Weights/"
    weight = '_best_val'

    ### load in data
    inputs, outputs, shape  = DF.load_ABM_data(path+file_name+".npy",plot=False)
    x = inputs[:,0]
    t = inputs[:,1]

    models = []
    binns = []

    for i in np.arange(5):

        binn_name  = f"{model_name}"
        save_name =  f"BINN_training_{binn_name}_training_replicate_{i}_{file_name}"
        model,binn = load_model(binn_name=binn_name,save_name=save_folder + save_name,x=x,t=t)

        models.append(model)
        binns.append(binn)
        
    
    
    
    x,t = unique_inputs(inputs)
    Dq = [Pm/4,Pinteraction]
    Dr = [100*Pp]

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
    shutil.copy(f"{save_folder}BINN_training_{binn_name}_training_replicate_{selected_replicate}_{file_name}{weight}_model",
                f"{save_folder}BINN_training_{binn_name}_{file_name}{weight}_model")