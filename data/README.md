Simulated ABM Data from the Pulling, Adhesion, and Pulling & Adhesion ABMs

Data from the Pulling ABM is stored in files named:
f"simple_pulling_mean_25_Pm_{rmp}_Ppull_{Ppull}.npy"
where rmp is the value of rmp and Ppull is the value of Ppull

Data from the Adhesion ABM is stored in files named:
f"simple_adhesion_mean_25_Pm_{rmh}_Padh_{Padh}.npy"
where rmh is the value of rmh and Padh is the value of Padh

Data from the Pulling & Adhesion ABM is stored in files named:
f"adhesion_pulling_mean_25_PmH_{rmh}_PmP_{rmp}_Padh_{Padh}_Ppull_{Ppull}_alpha_{alpha}.npy""
where rmh is the value of rmh, rmp is the value of rmp, Padh is the value of Padh, Ppull is the value of Ppull, and alpha is the value of alpha.

each file can be opened as a dictionary by entering:

data = np.load(filename, allow_pickle=True).item()

data is a dictionary. View its keys by entering data.keys(). Most importantly, data['C'] will return the average simulated ABM data.