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
groups = [
    # "attack-pbt-push-adv-1-1-op",
    "attack-pbt-push-adv-1-2-op",
    "attack-pbt-push-adv-1-4-op",
    "attack-pbt-push-adv-1-8-op",
    "attack-pbt-push-adv-1-16-op",
]
max = 50_000_000

df = pd.DataFrame()
for group in groups:
    new_df = get_run_df(group, log_name, max_step=max)
    df = df.append(new_df)


# sp = get_run_df("attack-sp-laser", log_name, max_step=max)

with setup_styles(["paper", "training-curve-2col"]):
    # ax = sns.lineplot(
    #     x="timestep",
    #     y="value",
    #     data=pd.melt(df, ["timestep"]),
    #     hue="variable",
    #     palette=["red"],
    #     legend=False,
    #     label="Self-play",
    # )

    ax = sns.lineplot(
        x="timestep",
        y="value",
        data=pd.melt(df, ["timestep"]),
        hue="variable",
        palette=["red"],
        legend=False,
        label="Adv. vs PBRL",
        # ax=ax,
    )

    ax.set(ylim=(-5, 5))
    ax.axhline(1.825040, color="black", linestyle="--", label="PBRL vs PBRL")
    # handles, _ = ax.get_legend_handles_labels()
    # handles = [handles[0], handles[2]]  # Remove the legend for the CI
    ax.legend(loc="lower right")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return")
    plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    # plt.show()
    plt.savefig("attack-push-pbt-curves.pdf")
