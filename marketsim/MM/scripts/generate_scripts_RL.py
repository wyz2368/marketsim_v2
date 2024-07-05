# Define the list of variables
game_name_list = ["TouchMM"]
root_result_folder_list = ["./root_result_RL"]
num_iteration_list = [2000]
num_background_agents_list = [25]
sim_time_list = [100000]
lam_list = [0.075]
lamMM_list = [0.005]
omega_list = [10]
K_list = [20]
n_levels_list = [21]
total_volume_list = [100]

#RL
epoch_list = [100]
step_per_epoch_list = [100000]
training_num_list = [1]
test_num_list = [1]


file_name = "run_static_RL.sh"

# Generate the bash script content
bash_script_content = ""

for game_name in game_name_list:
    for root_result_folder in root_result_folder_list:
        for num_iteration in num_iteration_list:
            for num_background_agents in num_background_agents_list:
                for sim_time in sim_time_list:
                    for lam in lam_list:
                        for lamMM in lamMM_list:
                            for omega in omega_list:
                                for K in K_list:
                                    for n_levels in n_levels_list:
                                        for total_volume in total_volume_list:
                                            for epoch in epoch_list:
                                                for step_per_epoch in step_per_epoch_list:
                                                    for training_num in training_num_list:
                                                        for test_num in test_num_list:
                                                            bash_script_content += (
                                                                f"python simMM_example.py --game_name={game_name} "
                                                                f"--root_result_folder={root_result_folder} "
                                                                f"--num_iteration={num_iteration} "
                                                                f"--num_background_agents={num_background_agents} "
                                                                f"--sim_time={sim_time} "
                                                                f"--lam={lam} "
                                                                f"--lamMM={lamMM} "
                                                                f"--omega={omega} "
                                                                f"--K={K} "
                                                                f"--n_levels={n_levels} "
                                                                f"--epoch={epoch} "
                                                                f"--step_per_epoch={step_per_epoch} "
                                                                f"--training_num={training_num} "
                                                                f"--test_num={test_num} "
                                                                f"--total_volume={total_volume} "
                                                            )


# Write the bash script to a file
with open(file_name, 'w') as file:
    file.write(bash_script_content)

print(f"Bash script generated and saved as '{file_name}'")
