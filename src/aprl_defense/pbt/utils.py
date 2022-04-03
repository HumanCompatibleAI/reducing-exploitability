import math
import random
from typing import Callable

import ray
import wandb
from ray.rllib.agents import Trainer
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
from ray.rllib.evaluation.worker_set import WorkerSet

from aprl_defense.common.base_logger import logger


def create_policy_mapping_function(main_agent_id, main_pol, ops: list) -> Callable:
    """Returns a callable policy mapping function that maps to either main pol or a random op from the ops list, depending on whether agent_id is main _id or
    not."""

    # This policy mapping function chooses random opponents from the list
    def map_fn(agent_id, episode, **kwargs):
        """Policy mapping function that deterministically chooses based on worker id and episode id"""
        if agent_id == main_agent_id:
            return main_pol
        else:  # The agent that is controlled by one of the opponent agents
            op_policy = random.choice(ops)
            return op_policy

    return map_fn


def create_pbt_eval_func(eval_agents: dict):
    main_name = "main"
    secondary_name = "eval_op"

    # Define the function
    def pbt_eval_function(trainer: Trainer, eval_workers: WorkerSet):
        """Custom evaluation function for PBT.
        Args:
            trainer (Trainer): trainer class to evaluate.
            eval_workers (WorkerSet): evaluation workers.
        Returns:
            metrics (dict): evaluation metrics dict.
        """

        remote_workers = eval_workers.remote_workers()

        my_eval_line_plot = {}
        metrics = {}

        for key, val in eval_agents.items():
            # Val is a list where each index corresponds to an op id
            # For every op id we either have the weights, or a list of historical weights
            if isinstance(val, list):
                if len(val) > 0:
                    # If we have a list of historical generations we iterate through these
                    for i, generation in enumerate(val):
                        num_opponents = len(generation)
                        metrics = _evaluate_metrics_helper_separate(
                            num_opponents, remote_workers, trainer, generation
                        )

                        reward_means = metrics["policy_reward_mean"]

                        if main_name in reward_means:
                            my_eval_line_plot[f"eval_{key}_{i}"] = reward_means[
                                main_name
                            ]

                            if (
                                i == len(val) - 1
                            ):  # We also log the newest generation separately in a line connecting all the newest evals
                                my_eval_line_plot[f"eval_{key}_newest"] = reward_means[
                                    main_name
                                ]
            else:
                num_opponents = len(val)
                metrics = _evaluate_metrics_helper_separate(
                    num_opponents, remote_workers, trainer, [val]
                )

                reward_means = metrics["policy_reward_mean"]

                if main_name in reward_means:
                    my_eval_line_plot[f"eval_{key}"] = metrics["policy_reward_mean"][
                        main_name
                    ]

        # TODO: Should we also log separate metrics for each opponent_id?

        metrics["my_eval_line_plot"] = my_eval_line_plot

        return metrics

    def _evaluate_metrics_helper_separate(
        num_opponents, remote_workers, trainer, weights_list: list
    ):
        """Helper function that runs eval workers for all agents that can be received from the function.
        weight_func is lambda that receives the opponent index and returns the opponent weights that are supposed to be evaluated."""
        num_workers = len(remote_workers)

        # If num_workers < num_agents we need to do multiple passes of eval
        num_eval_passes = math.ceil(num_opponents / num_workers)

        all_eps = []
        for pass_i in range(num_eval_passes):
            opponent_i = pass_i * num_workers
            # Number of workers to use this pass, either num_workers or the number of agents that are left
            num_workers_this_pass = (
                num_workers
                if opponent_i + num_workers <= num_opponents
                else num_opponents - opponent_i
            )

            worker_slice = remote_workers[
                :num_workers_this_pass
            ]  # Iterate over first n_w_t_p workers only

            # Set the weights for the workers
            for worker in worker_slice:
                if opponent_i < num_opponents:
                    weights = list(weights_list[opponent_i].values())[0]
                    worker.set_weights.remote(({secondary_name: weights}))
                    opponent_i += 1

            # Run the eval episodes
            for _ in range(trainer.config["evaluation_duration"]):
                # Calling .sample() runs exactly one episode per worker due to how the
                # eval workers are configured.
                ray.get([w.sample.remote() for w in worker_slice])

            # Collect the accumulated episodes on the workers
            episodes, _ = collect_episodes(
                remote_workers=worker_slice, timeout_seconds=99999
            )  # TODO timeout? what's up with that

            all_eps += episodes
        # All eps collected for this set of agents

        # Summarize the episode stats into a metrics dict
        # You can compute metrics from the episodes manually, or use the
        # convenient `summarize_episodes()` utility:
        metrics = summarize_episodes(all_eps)
        # Note that the above two statements are the equivalent of:
        # metrics = collect_metrics(None, worker_slice)

        # Some sanity checks
        if (
            metrics["episodes_this_iter"]
            != trainer.config["evaluation_duration"] * num_opponents
        ):
            logger.warning(
                f"num_workers_this_pass:{num_workers_this_pass}\n"
                f"worker_slice:{worker_slice}\n"
                f"num_workers:{num_workers}\n"
                f"num_opponents:{num_opponents}"
            )
        return metrics

    # Return it
    return pbt_eval_function


def custom_eval_log(results, timesteps_total, timesteps_main):
    # Log evaluation results if applicable
    if "evaluation" in results and "my_eval_line_plot" in results["evaluation"]:
        for key, val in results["evaluation"]["my_eval_line_plot"].items():
            wandb.log(
                {key: val, "timestep_agg": timesteps_total, "timestep": timesteps_main}
            )
