# DATA

In this folder, all the data is stored and handled. 
This includes the private data (not publicly shared), 
the aggregated information from the private data 
(publicly shared and used as input for the simulator), 
and the transaction logs of the simulator.

#### Data handling

- The original (private!) data is in  the folder 
data/real_data/transaction_log.csv. 
It can be produced from the raw data (anonymized_dataset.csv) 
with preprocess_data_raw.py.

- Aggregated information from the original data is stored 
in the folder data/simulator_input. This is used 
as direct input to the simulator. The data can be produced 
by running the ipython notebook 'analyse_data.ipynb'

- The simulator saves the logs in 
data/simulator_log/transaction_log.csv.