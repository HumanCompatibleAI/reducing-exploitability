import random


def random_scheduler(agent_infos):
    id = random.randint(0, agent_infos["num_agents"] - 1)
    return id


def random_choice_scheduler(agent_infos):
    choices = []
    for i in range(agent_infos["num_agents"]):
        if not agent_infos["deactivated"][i]:
            choices.append(i)
    id = random.choice(choices)
    return id


def random_policy_scheduler(agent_infos):
    op_policy = random.choice(agent_infos["opponent_policies"])

    return op_policy
