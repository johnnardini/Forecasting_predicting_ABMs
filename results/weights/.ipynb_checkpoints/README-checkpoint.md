Weights from training BINNs to ABM data

Weight data from training to the Pulling ABM is stored in files named:
f"SF_only_DMLP_training_replicate_{i}_simple_pulling_mean_25_Pm_{rmp}_Ppull_{ppull}_pde_weight_10000.0_best_val_model"
f"BINN_training_DMLP_training_replicate_{i}_simple_pulling_mean_25_Pm_{rmp}_Ppull_{ppull}_pde_weight_10000.0_best_val_model"
where i is the BINN training replicate, rmp is the value of rmp and Ppull is the value of Ppull. Data with No replicate is the final selected BINN model from the 5 replicates

The first file (SF_only*) corresponds to the first phase of BINN model training, where TMLP is trained to match the ABM data. The second file (BINN_training*) corresponds to the second phase of BINN model training, where TMLP is trained to match the ABM data, while TMLP and DMLP are both trained to satisfy the diffusion PDE framework.


Weight data from training to the Adhesion ABM is stored in files named:
f"SF_only_DMLP_training_replicate_{i}_simple_adhesion_mean_25_Pm_{rmh}_Padh_{padh}_pde_weight_10000.0_best_val_model"
f"BINN_training_DMLP_training_replicate_{i}_simple_adhesion_mean_25_Pm_{rmh}_Padh_{padh}_pde_weight_10000.0_best_val_model"
where i is the BINN training replicate, rmh is the value of rmh, and Padh is the value of Padh. Data with No replicate is the final selected BINN model from the 5 replicates

Weight data from training to the Pulling & Adhesion ABM is stored in files named:
f"SF_only_DMLP_training_replicate_{i}_adhesion_pulling_mean_25_PmH_{rmh}_PmP_{rmp}_Padh_{padh}_Ppull_{ppull}_alpha_{alpha}_pde_weight_10000.0_best_val_model"
f"BINN_training_DMLP_training_replicate_{i}_adhesion_pulling_mean_25_PmH_{rmh}_PmP_{rmp}_Padh_{padh}_Ppull_{ppull}_alpha_{alpha}_pde_weight_10000.0_best_val_model"
where i is the BINN training replicate, rmh is the value of rmh, rmp is the value of rmp, Padh is the value of Padh, Ppull is the value of Ppull, and alpha is the value of alpha.

Weight data stored at "filename" can be imported into memory with the following commands (x and t are the 1d grids of ABM data that the BINN was trained against)

from src.custom_functions import load_model
save_folder = ../../results/weights/
model,binn = load_model(binn_name="DMLP",save_name=save_folder + filename,x=x,t=t)
