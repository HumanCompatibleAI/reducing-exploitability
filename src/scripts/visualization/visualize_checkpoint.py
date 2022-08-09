from pathlib import Path

from aprl_defense.eval import eval_checkpoint

result = eval_checkpoint(
    agent_alg="ppo",
    artifact_identifier="diox6jdz:latest",
    num_steps=10000,
    render=True,
    local_mode=False,
    wandb_project="pbrl-defense",
    wandb_group="eval",
    # scenario_name='SumoHumans-v0',
    # scenario_name='YouShallNotPassHumans-v0',
    scenario_name="simple_push",
    video_dir=None,
    env_name="mpe",
    artifact_dir=Path("/home/pavel/code/chai/out/tmp"),
)
print(result)
