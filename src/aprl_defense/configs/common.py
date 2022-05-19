offline_eval_config = {
    # --- Parallelism ---
    "num_workers": 3,
    "num_gpus": 0,
    "num_gpus_per_worker": 0,
    "horizon": 25,  # !IMPORTANT! Without this plotting results won't work
    "multiagent": {  # IMPORTANT, overwrite these settings before using config
        "policies": {},
        "policy_mapping_fn": None,
    },
    "env_config": {
        "scenario_name": None,  # IMPORTANT, overwrite
    },
    # See explanation in train config
    "normalize_actions": False,
}
