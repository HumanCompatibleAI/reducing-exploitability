from aprl_defense.eval import multi_eval

# TODO: remove, once this file is updated
# pytype: skip-file

checkpoints_folder_main = "/home/pavel/zs_ray_results/many_train1/"
checkpoints_folder_other = "/home/pavel/zs_ray_results/new-ppo/"

main_alg = "maddpg"
other_alg = "ppo"

main_folders = [main_alg + "/01", main_alg + "/02"]
other_folders = ["sgd-iter-4/02/brpq184s", "sgd-iter-4/01/26e8baev"]
num_steps = 100000
local_mode = False

print(f"==== Just {main_alg} ====")

results_mm = multi_eval(
    agent_algs=[main_alg, main_alg],
    agent_paths=[
        checkpoints_folder_main + main_folders[0],
        checkpoints_folder_main + main_folders[1],
    ],
    num_steps=num_steps,
    render=False,
    local_mode=local_mode,
)

print(f"==== Just {other_alg} ====")

results_other_other = multi_eval(
    agent_algs=[other_alg, other_alg],
    agent_paths=[
        checkpoints_folder_other + other_folders[0],
        checkpoints_folder_other + other_folders[1],
    ],
    num_steps=num_steps,
    render=False,
    local_mode=local_mode,
)

print(f"==== {main_alg} v {other_alg} ====")

results_main_other = multi_eval(
    agent_algs=[main_alg, other_alg],
    agent_paths=[
        checkpoints_folder_main + main_folders[0],
        checkpoints_folder_other + other_folders[1],
    ],
    num_steps=num_steps,
    render=False,
    local_mode=local_mode,
)

print(f"==== {other_alg} v {main_alg} ====")

results_other_main = multi_eval(
    agent_algs=[other_alg, main_alg],
    agent_paths=[
        checkpoints_folder_other + other_folders[0],
        checkpoints_folder_main + main_folders[1],
    ],
    num_steps=num_steps,
    render=False,
    local_mode=local_mode,
)

print("==== FINAL RESULTS ====")

print(f"num_steps: {num_steps}")

print(f"{main_alg} v {main_alg}: {results_mm}")

print(f"{other_alg} v {other_alg}: {results_other_other}")

print(f"{main_alg} v {other_alg}: {results_main_other}")

print(f"{other_alg} v {main_alg}: {results_other_main}")
