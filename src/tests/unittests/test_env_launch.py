import unittest
from pathlib import Path

import ray
import wandb
from pyspiel import SpielError

from aprl_defense.common.artifact_manager import ArtifactManager
from aprl_defense.train import train

# TODO: update the file so this is not necessary anymore
# pytype: skip-file


def _start_train_with_env(env):
    path = Path("/tmp/aprl_defense_unittest")
    mode = "normal"
    artifact_manager = ArtifactManager(
        save_remote=False,
        save_locally=False,
        local_checkpoint_dir=path,
    )
    artifact_manager.init_saving_checkpoints(mode, env_name=env, metadata={})

    train(
        mode=mode,
        out_path=path,
        alg="ppo",
        timesteps=8000,
        checkpoint_freq=4000,
        env=env,
        artifact_manager=artifact_manager,
    )

    ray.shutdown()


def _start_pbt_with_env(env):
    path = Path("/tmp/aprl_defense_unittest")
    mode = "single-trainer-pbt"
    artifact_manager = ArtifactManager(
        save_remote=False,
        save_locally=False,
        local_checkpoint_dir=path,
    )
    artifact_manager.init_saving_checkpoints(mode, env, {})

    train(
        mode=mode,
        out_path=path,
        alg="ppo",
        num_ops=2,
        timesteps=8000,
        checkpoint_freq=4000,
        env=env,
        artifact_manager=artifact_manager,
    )

    ray.shutdown()


class TestConfig(unittest.TestCase):
    def setUp(self):
        # In order for the code to work we must init wandb. However, we set the mode to 'disabled' so nothing is actually logged
        wandb.init(mode="disabled")  # Disable wandb for these tests

    def tearDown(self):
        wandb.finish()

    def test_mpe_envs_train_normal(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        implemented_envs = [
            "mpe_simple_push",
            "mpe_simple_push_comm",
        ]

        for env in implemented_envs:
            print(f"Testing {env}")
            try:
                _start_train_with_env(env)
            except NotImplementedError:
                self.fail(
                    f"Unexpectedly raised NotImplementedError for env {env} which should be implemented!"
                )

    def test_os_envs_train_normal(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        implemented_envs = ["os_markov_soccer", "os_laser_tag"]

        for env in implemented_envs:
            print(f"Testing {env}")
            try:
                _start_train_with_env(env)
            except NotImplementedError:
                self.fail(
                    f"Unexpectedly raised NotImplementedError for env {env} which should be implemented!"
                )

    def test_pz_envs_train_normal(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        implemented_envs = ["pz_rps"]

        for env in implemented_envs:
            print(f"Testing {env}")
            try:
                _start_train_with_env(env)
            except NotImplementedError:
                self.fail(
                    f"Unexpectedly raised NotImplementedError for env {env} which should be implemented!"
                )

    def test_unsupported_envs_train_normal(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        not_implemented_envs = [
            "abcdefg",
            "abc_foobar",
        ]
        not_implemented_scenarios = [
            "mpe_foobar",
            "os_raboof",
            "pz_connect_four",  # Not implemented yet, might implement in future
        ]
        for env in not_implemented_envs:
            self.assertRaises(ValueError, _start_train_with_env, env)
            ray.shutdown()
        for env in not_implemented_scenarios:
            self.assertRaises(
                (FileNotFoundError, SpielError, NotImplementedError),
                _start_train_with_env,
                env,
            )
            ray.shutdown()

    def test_mpe_envs_pbt(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        implemented_envs = [
            "mpe_simple_push",
            "mpe_simple_push_comm",
        ]

        for env in implemented_envs:
            print(f"Testing {env}")
            try:
                _start_pbt_with_env(env)
            except NotImplementedError:
                self.fail(
                    f"Unexpectedly raised NotImplementedError for env {env} which should be implemented!"
                )

    def test_os_envs_pbt(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        implemented_envs = ["os_markov_soccer", "os_laser_tag"]

        for env in implemented_envs:
            print(f"Testing {env}")
            try:
                _start_pbt_with_env(env)
            except NotImplementedError:
                self.fail(
                    f"Unexpectedly raised NotImplementedError for env {env} which should be implemented!"
                )

    def test_pz_envs_pbt(self):
        """This checks that the implemented envs don't raise NotImplementedError.
        Furthermore, if exceptions are introduced for some envs, those will be raised here."""
        implemented_envs = ["pz_rps"]

        for env in implemented_envs:
            print(f"Testing {env}")
            try:
                _start_pbt_with_env(env)
            except NotImplementedError:
                self.fail(
                    f"Unexpectedly raised NotImplementedError for env {env} which should be implemented!"
                )
