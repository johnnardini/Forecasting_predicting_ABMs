MPI_data_generation.py is used to simulate and save data from all ABMs in this study. You can run this file from the terminal or
from a shell script with the command

```
mpirun python MPI_data_generation.py
```

The ABM that is simulated in the code depends on the variable `model_name`<br> (see lines 21-26)
    - When `model_name` = "simple_pulling", we generate ABM simulations for forecasting the Pulling ABM<br>
    - When `model_name` = "simple_adhesion", we generate ABM simulations for forecasting the Adhesion ABM<br>
    - When `model_name` = "adhesion_pulling", we generate ABM simulations for forecasting the Pulling & Adhesion ABM<br>
    - When `model_name` = "simple_adhesion_Padh_interp", we generate ABM simulations for predicting the Adhesion ABM as Padh varies and Pm is fixed<br>
    - When `model_name` = "simple_adhesion_Pm_Padh_interp", we generate ABM simulations for predicting the Adhesion ABM as Pm and Padh vary<br>
    - When `model_name` = "adhesion_pulling_LHC", we generate ABM simulations for predicting the Pulling & Adhesion ABM as rmH and rmP are fixed and Padh, Ppull, and alpha are varied.