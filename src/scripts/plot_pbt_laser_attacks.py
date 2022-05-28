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
    "attack-sp-laser",
    "attack-pbt-laser-20-v2",
    "attack-pbt-laser-40-v2",
    "attack-pbt-laser-60-v2",
    "attack-pbt-laser-80-v2",
]
names = ["Self-play", "PBRL-20", "PBRL-40", "PBRL-60", "PBRL-80"]
max = 50_000_000

with setup_styles(["paper", "training-curve-1col-tall-legend"]):
    for group, name in zip(groups, names):
        df = get_run_df(group, log_name, max_step=max)

        ax = sns.lineplot(
            x="timestep",
            y="value",
            data=pd.melt(df, ["timestep"]),
            # hue="variable",
            # palette=["red"],
            ci=False,
            legend=False,
            label=name,
        )

    ax.set(ylim=(-500, 500))
    ax.axhline(0, color="black", linestyle="--", label="Zero")

    ax.legend(
        title="Victim", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Return")
    # fig = plt.figure(figsize=(10, 5))
    plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    # plt.show()
    plt.savefig("attack-laser-pbt-curves.pdf")
