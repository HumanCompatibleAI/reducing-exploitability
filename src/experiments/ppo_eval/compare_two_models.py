from aprl_defense.eval import multi_eval

checkpoints_folder_main = "/scratch/pavel/out/"

main_alg = "ppo"

main_folders = ["1dtxv4w2", "7-fixed-attack/2o3jr4ke"]
num_steps = 100000
local_mode = False


print(f"folders: {main_folders}")

text_a = "victim vs adversary"
print(f"==== {text_a} ====")

results_ab = multi_eval(
    agent_algs=[main_alg, main_alg],
    agent_paths=[
        checkpoints_folder_main + main_folders[0],
        checkpoints_folder_main + main_folders[1],
    ],
    agent_to_policy_name=["policy_0", "policy_1"],
    num_steps=num_steps,
    render=False,
    local_mode=local_mode,
)

text_b = "victim self-play op"
print(f"==== {text_b} ====")

results_ba = multi_eval(
    agent_algs=[main_alg, main_alg],
    agent_paths=[
        checkpoints_folder_main + main_folders[0],
        checkpoints_folder_main + main_folders[0],
    ],
    agent_to_policy_name=["policy_0", "policy_1"],
    num_steps=num_steps,
    render=False,
    local_mode=local_mode,
)

print("==== FINAL RESULTS ====")

print(f"folders: {main_folders}")

print(f"num_steps: {num_steps}")

print(f"{text_a}: {results_ab}")

print(f"{text_b}: {results_ba}")
