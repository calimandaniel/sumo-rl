import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents.qlearning_agent import QLAgent
from sumo_rl.exploration import EpsilonGreedy

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    alpha = 0.001
    gamma = 0.5
    decay = 0.999
    runs = 1
    episodes = 1

    name = "qlearning_2x2_pressure"

    env = SumoEnvironment(
        net_file="./nets/2x2grid/2x2.net.xml",
        route_file="./nets/2x2grid/2x2.rou.xml",
        use_gui=False,
        num_seconds=500000,
        min_green=5,
        delta_time=5,
        reward_fn="diff-waiting-time"
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: QLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=1, min_epsilon=0.05, decay=decay),
            ) 
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(f"outputs/{name}/ql-2x2_run{run}", episode)
        QLAgent.save_all(ql_agents, name)
            # Assuming 'ql_agents' is your dictionary of agents
    for agent_id, agent in ql_agents.items():
        plt.plot(agent.rewards, label=f'Agent {agent_id}')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    plt.legend()
    # Save the plot before showing it
    plt.savefig('reward_qlearning.png')
    plt.show()
    
    env.close()
