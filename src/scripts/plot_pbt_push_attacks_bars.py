import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.common import get_run_df
from scripts.stylesheets import setup_styles



log_name = "adversary_reward"
groups = [
    "attack-sp-push-adv-1",
    # "attack-pbt-push-adv-1-1-op",
    "attack-pbt-push-adv-1-2-op",
    "attack-pbt-push-adv-1-4-op",
    "attack-pbt-push-adv-1-8-op",
    "attack-pbt-push-adv-1-16-op",
]
names = ["Self-\nplay", #"PBRL\n1",
         "PBRL\n2", "PBRL\n4", "PBRL\n8", "PBRL\n16"]
# Factor 4 for self-play because it has 4 times the per-policy batch_size
max = 25_000_000
min = 24_750_000

df = pd.DataFrame()

data = []
for group, name in zip(groups, names):
    new_df = get_run_df(group, log_name, min_step=min, max_step=max)
    data.append(new_df["return"].values)

# from_dict = pd.DataFrame.from_dict(data)

with setup_styles(["paper", "barplot-2col-bigger"]):
    palette = sns.color_palette("bright")
    clrs = ["red", "limegreen", "limegreen", "limegreen", "limegreen"]
    ax = sns.barplot(data=data, palette=clrs, capsize=0.2)

    # Thresholds
    thresholds = [
        1.783328,  # Selfplay
        # 1.415797e-01,  # PBT-1
        1.825040,
        1.825040,
        1.825040,
        1.825040,
    ]

    for i, a in enumerate(ax.patches):
        x_start = a.get_x()
        width = a.get_width()
        ax.plot([x_start, x_start + width], 2 * [thresholds[i]], "--", c="k")

    # ax.axhline(1.783328, color="black", linestyle="--", label="Zero")
    # num_op 1: 1.415797e-01
    #
    # ax.legend(loc="lower right")
    ax.set_xlabel("Victim")
    ax.set_ylabel("Return")
    # # fig = plt.figure(figsize=(10, 5))
    # plt.xticks([0, 1e7, 2e7, 3e7, 4e7, 5e7])
    plt.xticks(range(len(names)), names)
    # plt.show()
    plt.savefig("attack-push-pbt-1.pdf")
