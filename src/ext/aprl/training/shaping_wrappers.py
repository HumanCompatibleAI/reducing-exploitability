import gym

from ext.aprl.training.scheduling import (
    ConditionalAnnealer,
    ConstantAnnealer,
    LinearAnnealer,
    get_annealer_from_actor,
)

REW_TYPES = set(("sparse", "dense"))


class RewardShapingVecWrapper(gym.Wrapper):
    """
    A more direct interface for shaping the reward of the attacking agent.
    - shaping_params schema: {'sparse': {k: v}, 'dense': {k: v}, **kwargs}
    """

    def __init__(self, venv, shaping_params, reward_annealer=None):
        super().__init__(venv)
        assert shaping_params.keys() == REW_TYPES
        self.shaping_params = {}
        for rew_type, params in shaping_params.items():
            for rew_term, weight in params.items():
                self.shaping_params[rew_term] = (rew_type, weight)

        self.reward_annealer = reward_annealer
        c = self.reward_annealer()
        self.cache = [c]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, infos = self.env.step(action)
        rew = list(rew)  # Convert from tuple to list to allow for in-place modification
        for agent_idx in range(len(obs)):
            # Compute shaped_reward for each rew_type
            shaped_reward = {k: 0 for k in REW_TYPES}
            for rew_term, rew_value in infos[agent_idx].items():
                if rew_term not in self.shaping_params:
                    continue
                rew_type, weight = self.shaping_params[rew_term]
                shaped_reward[rew_type] += weight * rew_value

            # Compute total shaped reward, optionally annealing
            rew[agent_idx] = _anneal(
                shaped_reward, self.reward_annealer, cache=self.cache
            )

            logged_agent_rewards = {}
            logged_agent_rewards.update(shaped_reward)
            logged_agent_rewards["reward"] = rew[agent_idx]

            infos[agent_idx]["logged_agent_rewards"] = logged_agent_rewards

            c = self.reward_annealer(cache=self.cache)
            infos[agent_idx]["dense_weight"] = c

        return obs, rew, done, infos


def apply_reward_wrapper(single_env, shaping_params, scheduler):
    if "metric" in shaping_params:
        rew_shape_annealer = ConditionalAnnealer.from_dict(
            shaping_params, get_logs=None
        )
        scheduler.set_conditional.remote("rew_shape")
    else:
        anneal_frac = shaping_params.get("anneal_frac")
        if anneal_frac is not None:
            rew_shape_annealer = LinearAnnealer(1, 0, anneal_frac)
        else:
            # In this case, we weight the reward terms as per shaping_params
            # but the ratio of sparse to dense reward remains constant.
            rew_shape_annealer = ConstantAnnealer(0.5)

    scheduler.set_annealer.remote("rew_shape", rew_shape_annealer)
    return RewardShapingVecWrapper(
        single_env,
        shaping_params=shaping_params["weights"],
        reward_annealer=get_annealer_from_actor(scheduler, "rew_shape"),
    )


def _anneal(reward_dict, reward_annealer, cache=None):
    c = reward_annealer(cache=cache)
    assert 0 <= c <= 1
    sparse_weight = 1 - c
    dense_weight = c
    return reward_dict["sparse"] * sparse_weight + reward_dict["dense"] * dense_weight
