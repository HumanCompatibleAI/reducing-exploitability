offline_eval_update = {
    # --- Parallelism ---
    "num_workers": 4,
    "num_envs_per_worker": 1,
    # "num_gpus": 0,
    # "num_gpus_per_worker": 0,
    # "horizon": 25,  # !IMPORTANT! Without this plotting results won't wor
    # Normalize actions is not available for multi-agent envs, at least according to this: https://github.com/ray-project/ray/issues/8518k
    # For me this problem only occured with SAC, DDPG
    # "normalize_actions": False,
    # During training these values might be significantly smaller, for eval it should be fine to have larger values here
    # "train_batch_size": 1000,
    # "num_envs_per_worker": 4,
    # "sgd_minibatch_size": 128,
}

# These settings are for the online evaluation during training, currently used by pbt
online_eval = {
    # Number episodes each time evaluation runs.
    "evaluation_duration": 10,
    "evaluation_num_workers": 10,
    # Setting this to True is experimental
    "evaluation_parallel_to_training": True,
}
