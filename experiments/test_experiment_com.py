import argparse
import os
import sys

import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl.environment.env_com import SumoEnvironment
from sumo_rl.agents.dqn_agent import DQNAgent
from sumo_rl.exploration import EpsilonGreedy
from sumo_rl.agents.shared_q_net import SharedQNetwork

import torch


if __name__ == "__main__":
    alpha = 0.01
    gamma = 0.99
    decay = 0.9999
    runs = 1
    episodes = 1

    name = "neighbor_try_1"

    env = SumoEnvironment(
        net_file="nets/2x2grid/2x2.net.xml",
        route_file="nets/2x2grid/2x2.rou.xml",
        use_gui=False,
        num_seconds=1000,
        min_green=5,
        delta_time=5,
        reward_fn="queue"
    )

        

    for run in range(1, runs + 1):
        initial_states = env.reset()
        shared_q_net = SharedQNetwork(18*3, env.action_space.n) # 18 is the size of the observation space, 3 is the number of neighbors
        ql_agents = {}

        for ts in env.ts_ids:
            neighbors = env.traffic_signals[ts].get_neighbors()
            neighbors_obs = [env.encode(env.observations[neighbor], neighbor) for neighbor in neighbors if neighbor in env.observations]
            init_states = torch.cat((torch.tensor(neighbors_obs), torch.tensor(env.encode(initial_states[ts], ts)).unsqueeze(0)), dim=0).flatten()
            ql_agents[ts] = DQNAgent(
            id=ts,
            starting_state=init_states,
            state_space=env.observation_space,
            action_space=env.action_space,
            q_net=shared_q_net,
            alpha=alpha,
            gamma=gamma,
            exploration_strategy=EpsilonGreedy(initial_epsilon=1, min_epsilon=0.05, decay=decay),
            )
        for agent in ql_agents:
            print("AGENT: ", agent)
        

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = init_states

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    neighbors = env.traffic_signals[agent_id].get_neighbors()
                    neighbors_obs = [env.encode(env.observations[neighbor], neighbor) for neighbor in neighbors if neighbor in env.observations]
                    ql_agents[agent_id].learn(next_state=torch.cat((torch.tensor(env.encode(s[agent_id], agent_id)).unsqueeze(0), torch.tensor(neighbors_obs)), dim=0).flatten(), reward=r[agent_id])

            env.save_csv(f"outputs/{name}/ql-2x2_run{run}", episode)
        shared_q_net.save(name)

    env.close()
