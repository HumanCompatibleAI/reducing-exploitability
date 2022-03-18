from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv


class OpenSpielZSEnv(OpenSpielEnv):
    """This class wraps rllib's open spiel env wrapper, making it so the reward is zero sum. Currently only supports 2-agent envs."""

    def step(self, action):
        obs, rewards, dones, infos = super().step(action)

        assert (
            len(rewards) == 2
        ), "OpenSpielZSEnv only supports envs with exactly 2 agents"

        items_list = list(rewards.items())
        ag_0, rew_0 = items_list[0]
        ag_1, rew_1 = items_list[1]

        rewards[ag_0] -= rew_1
        rewards[ag_1] -= rew_0

        # Zero-sum property
        assert rewards[ag_0] == -rewards[ag_1]

        return obs, rewards, dones, infos
