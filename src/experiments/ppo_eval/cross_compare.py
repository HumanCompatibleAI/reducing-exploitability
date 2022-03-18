import numpy as np

from aprl_defense.eval import multi_eval

checkpoints_folder_main = "/scratch/pavel/out/9-comm/"

main_alg = "ppo"

scenario_name = "simple_push_comm"

main_folders = [  # Simple push with communication on rnn
    "34aphnzq",
    "2uhlxvzu",
    "3h90dkz7",
    "29wzrlx7",
    "2xafftpw",
    "1oct0lga",
    "3b4sol4r",
    "5hwd3qsd",
    "2qvmbqhu",
    "1eefhkpm",
]

# main_folders = ['2qjb9qso', # Normal simple push svm
#                 '1vqsnfzs',
#                 '2nqomi4d',
#                 '28p7l83y',
#                 '3juxjbos',
#                 '3spmkqei',
#                 '26m8i5ur',
#                 '27bvd59k',
#                 '1k92hc8r',
#                 '27bg537a'
#                 ]
num_steps = 100000
local_mode = False

print(f"folders: {main_folders}")

rewards_0 = []
rewards_1 = []

runs = 0
for folder_0 in main_folders:
    for folder_1 in main_folders:
        if folder_0 != folder_1:
            runs += 1
            results = multi_eval(
                agent_algs=[main_alg, main_alg],
                agent_paths=[
                    checkpoints_folder_main + folder_0,
                    checkpoints_folder_main + folder_1,
                ],
                agent_to_policy_name=["policy_0", "policy_1"],
                num_steps=num_steps,
                render=False,
                local_mode=local_mode,
                scenario_name=scenario_name,
            )
            rewards_0.append(results["mean_rewards"][0])
            rewards_1.append(results["mean_rewards"][1])

print(f"Performed {runs} runs")

mean_0 = np.mean(rewards_0)
mean_1 = np.mean(rewards_1)

std_0 = np.std(rewards_0)
std_1 = np.std(rewards_1)

print("===== FINAL RESULTS =====")

print(f"mean_0: {mean_0:.3f}")
print(f"mean_1: {mean_1:.3f}")
print(f"std_0: {std_0:.3f}")
print(f"std_1: {std_1:.3f}")
