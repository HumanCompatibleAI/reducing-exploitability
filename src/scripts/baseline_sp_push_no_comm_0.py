import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.common import get_run_df
from scripts.stylesheets import setup_styles

# run = api.run("chaiberkeley/pbrl-defense-icml/seqb62s8")
#
# print(run.config)
# print(run.history())
# print(run.summary)

log_name = "policy_0_reward"
group = "selfplay-sp-no-comm"
min = 24_750_000
max = 25_000_000

df = get_run_df(group, log_name, min, max)

print(f"For {log_name}")
print(f"Mean reward: {df.mean(axis=0)}")
print(f"STD: {df.std(axis=0)}")

print(f"For the other policy, take the negative")

# Return: -1.754714e+00
