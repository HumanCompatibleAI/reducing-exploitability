import contextlib
import pickle
from pathlib import Path
from typing import List, Union, cast

import fire
import ray
import wandb
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from ray.tune.utils import merge_dicts

import aprl_defense.configs.common
import aprl_defense.configs.eval
from aprl_defense.common.artifact_manager import ArtifactManager
from aprl_defense.common.in_memory_rollout_saver import InMemoryRolloutSaver
from aprl_defense.common.io import get_checkpoint_file
from aprl_defense.common.rllib_io import restore_trainer_from_path
from aprl_defense.common.utils import (
    trainer_cls_from_str,
    generate_multiagent_2_policies,
    noop_logger_creator,
    init_env,
)
from aprl_defense.evaluate import rollout


def multi_eval(
    agent_algs: List[str],
    agent_paths: Union[List[Path], List[str]],
    agent_to_policy_name: List[str],
    num_steps: int,
    render: bool,
    local_mode: bool,
    use_newest_checkpoint=True,
    scenario_name="simple_push",
) -> dict:
    num_agents = len(agent_algs)
    if num_agents != len(agent_paths):
        raise ValueError(
            f"Incompatible number of agents: len(agent_algs)={num_agents}, "
            f"len(agent_paths)={len(agent_paths)}"
        )
    if num_agents < 2:
        raise ValueError(f"Need at least 2 agents, got {num_agents}")

    if use_newest_checkpoint:
        for i, path in enumerate(agent_paths):
            checkpoint_file = get_checkpoint_file(path)

            agent_paths[i] = checkpoint_file

    trainer_classes = [trainer_cls_from_str(agent_alg) for agent_alg in agent_algs]

    new_policy_names = [f"policy_{i}" for i in range(num_agents)]

    ray.init(local_mode=local_mode)

    # Create and register env
    env = init_env(env_name="mpe", scenario_name=scenario_name, max_steps=25)

    # Load original weights by way of trainers
    restored_trainers = []  # Contains trainers for all agents except the first one
    for i in range(num_agents):
        trainer = restore_trainer_from_path(
            Path(agent_paths[i]),
            scenario_name,
            trainer_classes[i],
            config_update=aprl_defense.configs.eval.offline_eval_update,
        )
        restored_trainers.append(trainer)

    # Set up the config for the eval trainer
    eval_config = (
        aprl_defense.configs.common.offline_eval_config
    )  # get_victim_config(agent_paths[0])
    eval_config = _update_config_for_eval(eval_config, scenario_name)
    eval_config["multiagent"] = generate_multiagent_2_policies(
        env, policies_to_train=None
    )

    # Set policies according to loaded agents
    for i, trainer in enumerate(restored_trainers):
        policy = trainer.get_policy(agent_to_policy_name[i])
        policy_type = None if policy is None else type(policy)
        eval_config["multiagent"]["policies"][new_policy_names[i]] = (
            policy_type,
            env.observation_space_dict[i],
            env.action_space_dict[i],
            {"agent_id": i},
        )

    print(pretty_print(eval_config))

    # eval_config['evaluation_num_episodes'] = 4
    # Create the trainer to perform eval in
    eval_trainer = PPOTrainer(config=eval_config, env="current-env")

    # Set loaded weights
    for i, trainer in enumerate(restored_trainers):
        orig_pol_name = agent_to_policy_name[i]
        # Set the weights for the other policy
        eval_trainer.set_weights(
            {new_policy_names[i]: trainer.get_weights(orig_pol_name)[orig_pol_name]}
        )

    # Now the trainer object should have the agent0 weights in policy0 and agent1 weights in policy1

    # results = eval_trainer.evaluate()

    # ray.shutdown()
    #
    # return {
    #     'mean_rewards': results['policy_reward_mean'],
    #     'num_episodes': results['episodes_this_iter']
    # }
    saver = InMemoryRolloutSaver()

    # Suppress command line output because rollout() is a bit too verbose for my taste
    with contextlib.redirect_stdout(None):
        rollout(
            eval_trainer,
            None,  # Parameter is unused
            num_steps=num_steps,
            saver=saver,
            no_render=not render,
        )

    ray.shutdown()

    return {
        "mean_rewards": saver.mean_rewards,
    }


def eval_checkpoint(
    artifact_identifier: str,
    agent_alg: str = "ppo",
    render: bool = False,
    num_steps: int = 10000,
    wandb_project: str = "pbrl-defense",
    wandb_group: str = "dev",
    env_name="mpe",
    scenario_name="simple_push",
    artifact_dir: Union[str, Path] = Path("/scratch/pavel/out/tmp"),
    video_dir=None,
    local_mode: bool = False,
) -> dict:
    artifact_dir = Path(artifact_dir)
    wandb.init(project=wandb_project, group=wandb_group, dir=artifact_dir)

    ray.init(local_mode=local_mode)

    artifact_manager = ArtifactManager(
        save_remote=False,
        local_checkpoint_dir=artifact_dir.resolve() / wandb.run.id,
    )

    file = artifact_manager.get_remote_checkpoint(artifact_identifier)

    trainer_class = trainer_cls_from_str(agent_alg)

    env = init_env(env_name=env_name, scenario_name=scenario_name, max_steps=25)

    update_config = _update_config_for_eval(
        {
            "env_config": {"scenario_name": scenario_name},
            "num_workers": 0,
            "evaluation_num_workers": 0,
            "custom_eval_function": None,
            "evaluation_duration": 1,
        },
        scenario_name,
    )

    update_config["multiagent"] = generate_multiagent_2_policies(
        env, policies_to_train=None
    )

    trainer = restore_trainer_from_path(
        file, scenario_name, trainer_class, config_update=update_config
    )

    # Now the trainer object should have the agent0 weights in policy0 and agent1 weights in policy1
    saver = InMemoryRolloutSaver()

    # Suppress command line output because rollout() is a bit too verbose for my taste
    with contextlib.redirect_stdout(None):
        rollout(
            trainer,
            None,  # Parameter is unused
            num_steps=num_steps,
            saver=saver,
            no_render=not render,
            video_dir=video_dir,
        )

    ray.shutdown()

    return {
        "mean_rewards": saver.mean_rewards,
    }


def eval_from_mujoco_state(
    mujoco_state_path: str,
    agent_alg: str = "ppo",
    render: bool = False,
    num_steps: int = 10000,
    wandb_project: str = "pbrl-defense",
    wandb_group: str = "dev",
    env_name="mpe",
    scenario_name="simple_push",
    artifact_dir: Union[str, Path] = Path("/scratch/pavel/out/tmp"),
    video_dir=None,
    local_mode: bool = False,
) -> dict:
    artifact_dir = Path(artifact_dir)
    wandb.init(project=wandb_project, group=wandb_group, dir=artifact_dir)

    ray.init(local_mode=local_mode)

    trainer_class = trainer_cls_from_str(agent_alg)

    mujoco_state_file = Path(mujoco_state_path).open(mode="rb")
    mujoco_state = pickle.load(mujoco_state_file)

    _ = init_env(
        env_name=env_name,
        scenario_name=scenario_name,
        max_steps=25,
        mujoco_state=mujoco_state,
    )

    update_config = _update_config_for_eval(
        {
            "env_config": {"scenario_name": scenario_name},
            "num_workers": 0,
            "evaluation_num_workers": 0,
            "custom_eval_function": None,
            "evaluation_duration": 1,
        },
        scenario_name,
    )

    trainer = trainer_class(
        env="current-env", config=update_config, logger_creator=noop_logger_creator
    )
    # trainer = restore_trainer_from_path(file, scenario_name, trainer_class, config_update=update_config)

    # Now the trainer object should have the agent0 weights in policy0 and agent1 weights in policy1
    saver = InMemoryRolloutSaver()

    # Suppress command line output because rollout() is a bit too verbose for my taste
    with contextlib.redirect_stdout(None):
        rollout(
            trainer,
            None,  # Parameter is unused
            num_steps=num_steps,
            saver=saver,
            no_render=not render,
            video_dir=video_dir,
        )

    ray.shutdown()

    return {
        "mean_rewards": saver.mean_rewards,
    }


def _update_config_for_eval(config, scenario_name):
    """Updates the config for evaluation. Code from rllib/rollout.py"""

    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = cast(dict, config.get("evaluation_config", {}))
    evaluation_config["horizon"] = 25
    config = merge_dicts(config, evaluation_config)

    # Make sure we have evaluation workers.
    # if not config.get("evaluation_num_workers"):
    #     config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_duration"):
        config["evaluation_duration"] = 4
    config[
        "horizon"
    ] = 25  # Number of timesteps before the episode is forced to terminate

    config.update(aprl_defense.configs.eval.offline_eval_update)
    # config["render_env"] = not args.no_render
    # config["record_env"] = args.video_dir
    config["env_config"]["scenario_name"] = scenario_name  # IMPORTANT, overwrite

    return config


if __name__ == "__main__":
    fire.Fire(
        {
            "checkpoint": eval_checkpoint,
            "mujoco-state": eval_from_mujoco_state,
            "multi": multi_eval,
        }
    )
