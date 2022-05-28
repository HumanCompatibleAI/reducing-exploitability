import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.common import get_run_df
from scripts.stylesheets import setup_styles


log_name = "main"
group = "icml-pbt-push-v1"

num_opss = [1, 2, 4, 8, 16]
min = 24_750_000
max = 25_000_000

means = []
stds = []
for num_ops in num_opss:

    filter = {"main_id": 0, "num_ops": num_ops}

    df = get_run_df(group, log_name, min, max)
    means.append(df.mean(axis=0))
    stds.append(df.std(axis=0))

print(f"For {log_name}")
print(f"Mean reward: {means}")
print(f"STD: {stds}")
