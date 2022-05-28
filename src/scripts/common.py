from collections import defaultdict
from typing import Optional

import pandas as pd
import wandb


def get_run_df(
    group: str,
    log_name: str,  # e.g. 'adversary_reward', 'policy_0_reward', 'main'
    min_step: Optional[int] = None,
    max_step: Optional[int] = None,
    num_samples: Optional[int] = None,
    more_filter: Optional[dict] = None,
) -> pd.DataFrame:
    if num_samples is not None and max_step is None:
        raise ValueError("Must specify max_step if num_samples is specified.")

    api = wandb.Api()
    filter = {"config.wandb_group": group}
    if more_filter is not None:
        filter.update(more_filter)
    runs = api.runs(
        path="chaiberkeley/pbrl-defense-icml",
        filters=filter,
    )
    df = pd.DataFrame()
    counter = defaultdict(int)

    num_samples_for_run = [0] * len(runs)
    if num_samples is not None:
        samples_per_run = num_samples // len(runs)
        num_samples_for_run = [samples_per_run] * len(runs)
        # If it doesn't add up, distribute additional samples to runs
        for i in range(num_samples - samples_per_run * len(runs)):
            num_samples_for_run[i] += 1

    for i, run in enumerate(runs):
        print(run.summary)
        history = run.scan_history()
        values = []

        for row in history:
            if (
                log_name in row
                and (min_step is None or row["timestep"] > min_step)
                and (max_step is None or row["timestep"] < max_step)
            ):
                values.append((row["timestep"], row[log_name]))
                counter[row["timestep"]] += 1
        # In the case that we don't want to return a limited number of samples,
        # num_samples will be None and num_samples_for_run[i] will be 0.
        values = values[-num_samples_for_run[i] :]
        df = df.append(pd.DataFrame(values, columns=["timestep", f"return"]))
    counter_counter = defaultdict(int)
    for ts, count in counter.items():
        counter_counter[count] += 1
    # Sanity check
    print(counter_counter)

    return df
