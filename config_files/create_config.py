import os

# Define the directory and filename

config_dir = '/home/maltenageler/Studienarbeit/GP-SSM/config_files'
config_filename = 'my_config.ini' # Using the new filename
config_filepath = os.path.join(config_dir, config_filename)

# Create the directory if it doesn't exist
os.makedirs(config_dir, exist_ok=True)
# The detailed INI content from the 'pendulum_config_detailed' document
ini_content = """[DEFAULT]

# string values
data_path = /home/maltenageler/Studienarbeit/GP-SSM/Inverse Pendulum/pendulum_data/
file_name_1 = forced_pendulum_data_train.pkl
file_name_2 = forced_pendulum_data_test.pkl
file_name_M1 = forced_pendulum_data_train_mass_matrix.pkl
file_name_M2 = forced_pendulum_data_test_mass_matrix.pkl
model_loading_path = /home/maltenageler/Studienarbeit/GP-SSM/Inverse Pendulum/model/model_
saving_path = /home/maltenageler/Studienarbeit/GP-SSM/Results/pendulum/run_
dev_name = cuda:0
output_feature = output
noiseless_output_feature = output
robot_structure_str = pr

# boolean values
flg_load = True
flg_save = True
flg_save_trj = True
flg_train = False
flg_norm = False
flg_norm_noise = False
flg_mean_norm = False
single_sigma_n = True
shuffle = True
drop_last=True
flg_plot=True

# int values
num_dof = 2
num_prism = 1
num_rev = 1
num_dat_tr = -1
downsampling = 1
downsampling_data_load = 1
batch_size = 10
n_epoch = 301
n_epoch_print = 20
num_threads = 4

#float values (kept defaults, review if needed)
sigma_n_num = 0.0002
lr = 0.01
vel_threshold = -1

[LK_GIP]
LK_model_name = m_GP_LK_GIP
f_k_name = get_K_blocks_GIP_sum_InversePendulum_no_subs

[LK_GIP_sum]
lk_model_name = m_GP_LK_GIP_sum
f_k_name = get_K_blocks_GIP_sum_InversePendulum_no_subs
"""

# Write the content to the file
with open(config_filepath, 'w') as f:
    # .strip() removes leading/trailing whitespace from the multi-line string
    f.write(ini_content.strip())

print(f"Configuration file saved to: {config_filepath}")

# Optional: Create the other directories mentioned in the config
# These ensure the paths exist if your script expects them.
os.makedirs('./data', exist_ok=True)
os.makedirs('./Results/pendulum', exist_ok=True)

# You can verify the content by reading it back
# print("\n--- File Content ---")
# with open(config_filepath, 'r') as f:
#     print(f.read())
# print("--- End File Content ---")
