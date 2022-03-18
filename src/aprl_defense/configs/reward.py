default = {
    "weights": {"dense": {"reward_move": 0.1}, "sparse": {"reward_remaining": 0.01}},
    "anneal_frac": 0.5,
}

humanoid_stand = {
    "weights": {
        "dense": {
            "reward_linvel": 0,
            "reward_quadctrl": 0.1,
            "reward_alive": 0.1,
            "reward_impact": 0.1,
        },
        "sparse": {},
    }
}

humanoid = {
    "weights": {
        "dense": {
            "reward_linvel": 0.1,
            "reward_quadctrl": 0.1,
            "reward_alive": 0.1,
            "reward_impact": 0.1,
        },
        "sparse": {},
    }
}
