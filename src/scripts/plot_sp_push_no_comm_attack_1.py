import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.common import get_run_df
from scripts.stylesheets import setup_styles


log_name = "adversary_reward"
group = "attack-push-sp-no-comm"
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

    ax.set(ylim=(-5, 5))
    ax.axhline(1.754714, color="black", linestyle="--", label="Self-play")
    # handles, _ = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[2]]  # Remove the legend for the CI
    ax.legend(loc="lower right")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return")
    # plt.show()
    plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    plt.savefig("attack-push-no-comm-selfplay-1.pdf")
