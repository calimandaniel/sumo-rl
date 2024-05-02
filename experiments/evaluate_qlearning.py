import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents.qlearning_agent import QLAgent
from sumo_rl.exploration import EpsilonGreedy


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""Show trained junction"""
    )
    prs.add_argument("-s", type=int, default=1000, help="nr of simulated seconds\n")
    prs.add_argument("-g", type=int, default=5, help="Minimum green time\n")
    prs.add_argument("-n", required=True, type=str, help="Name of saved model")

    args = prs.parse_args()

    alpha = 0.001
    gamma = 0.99
    decay = 0.9999
    runs = 1
    episodes = 1

   

    env = SumoEnvironment(
        net_file="nets/2x2grid/2x2.net.xml",
        route_file="nets/2x2grid/2x2.rou.xml",
        use_gui=False,
        num_seconds=args.s,
        min_green=args.g,
        delta_time=5,
    )
    initial_states = env.reset()
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.05, decay=decay),
        ) 
        for ts in env.ts_ids
    }
    QLAgent.load_all(ql_agents, args.n)
    initial_states = env.reset()
    for ts in initial_states.keys():
        ql_agents[ts].state = env.encode(initial_states[ts], ts)
    infos = []
    done = {"__all__": False}
    while not done["__all__"]:
        actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

        s, r, done, info = env.step(action=actions)

        for agent_id in s.keys():
            ql_agents[agent_id].eval_step(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

    env.save_csv(f"outputs/evals/{args.n}/file", 1)
    

    env.close()

