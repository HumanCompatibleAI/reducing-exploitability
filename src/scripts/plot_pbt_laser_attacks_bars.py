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
names = ["Self-\nplay", "PBRL\n20", "PBRL\n40", "PBRL\n60", "PBRL\n80"]
# Factor 4 for self-play because it has 4 times the per-policy batch_size
max = 25_000_000
min = 24_750_000

df = pd.DataFrame()

data = []
for group, name in zip(groups, names):
    new_df = get_run_df(group, log_name, min_step=min, max_step=max)
    data.append(new_df["return"].values)

# from_dict = pd.DataFrame.from_dict(data)

with setup_styles(["paper", "barplot-2col"]):
    clrs = ["red", "red", "limegreen", "limegreen", "limegreen"]
    ax = sns.barplot(data=data, palette=clrs, capsize=0.2)
    # ax = sns.lineplot(
    #     x="timestep",
    #     y="value",
    #     data=pd.melt(df, ["timestep"]),
    #     # hue="variable",
    #     # palette=["red"],
    #     legend=False,
    #     label=name,
    # )
    # # ax.set(ylim=(-5, 5))
    ax.axhline(0, color="black", linestyle="--", label="Zero")
    #
    # ax.legend(loc="lower right")
    ax.set_xlabel("Victim")
    ax.set_ylabel("Return")
    # # fig = plt.figure(figsize=(10, 5))
    # plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    plt.xticks(range(len(names)), names)
    # plt.show()
    plt.savefig("attack-laser-pbt.pdf")
