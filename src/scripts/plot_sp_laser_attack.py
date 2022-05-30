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

log_name = "adversary_reward"
group = "attack-sp-laser"
max = 50_000_000

df = get_run_df(group, log_name, max_step=max)

with setup_styles(["paper", "training-curve-1col"]):
    ax = sns.lineplot(
        x="timestep",
        y="value",
        data=pd.melt(df, ["timestep"]),
        hue="variable",
        palette=["red"],
        legend=False,
        label="Adversary",
    )
    ax.set(ylim=(-750, 750))
    ax.axhline(0, color="black", linestyle="--", label="Zero")
    handles, _ = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[2]]  # Remove the legend for the CI
    ax.legend(loc="lower right", handles=handles)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return")
    plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    # plt.show()
    # fig = plt.figure(figsize=(10, 5))
    plt.savefig("attack-laser-selfplay.pdf")
