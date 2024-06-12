import argparse
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents.dqn_agent import DQNAgent
from sumo_rl.exploration import EpsilonGreedy
from sumo_rl.agents.shared_q_net import SharedQNetwork
from sumo_rl.agents.ddqn_agent import DoubleDQNAgent
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == "__main__":
    start_time = time.time()
    alpha = 0.001
    gamma = 0.5
    decay = 0.999
    runs = 1
    episodes = 1

    name = "best_params_2x2_pressure"

    env = SumoEnvironment(
        net_file="./nets/2x2grid/2x2.net.xml",
        route_file="./nets/2x2grid/2x2.rou.xml",
        use_gui=False,
        num_seconds=100000,
        min_green=5,
        delta_time=5,
        reward_fn="diff-waiting-time"
    )

    
    #alpha_values = [0.001, 0.01, 0.1]
    #gamma_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    #decay_values = [0.999, 0.99, 0.9]

    # In train_dqn.py
    # hidden_layers_values = [1, 2, 3, 4, 5, 6]
    # neurons_values = [64, 128]
    # use_ln_values = [False]
    # use_dropout_values = [False]
    
    
    hidden_layers_values = [3]
    neurons_values = [128]
    use_ln_values = [False]
    use_dropout_values = [False]
    best_reward = -float('inf')
    best_params = None
    # for gamma in gamma_values:
    for hidden_layers in hidden_layers_values:
        for neurons in neurons_values:
            for use_ln in use_ln_values:
                for use_dropout in use_dropout_values:
                    for run in range(1, runs + 1):
                        initial_states = env.reset()
                        shared_q_net = SharedQNetwork(18, env.action_space.n, hidden_layers, neurons, use_ln, use_dropout)
                        ql_agents = {
                            ts: DQNAgent(
                                id=ts,
                                starting_state=env.encode(initial_states[ts], ts),
                                state_space=env.observation_space,
                                action_space=env.action_space,
                                q_net=shared_q_net,
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
                            index = 0
                            while not done["__all__"]:
                                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                                s, r, done, info = env.step(action=actions)

                                for agent_id in s.keys():
                                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
                                
                                #if index % 100 == 0:
                                #    for agent in ql_agents.values():
                                #        agent.update_target_network()
                                #index += 1
                            
                            total_reward = sum(agent.acc_reward for agent in ql_agents.values())
                            print(f"Episode {episode} total reward: {total_reward}")

                        if total_reward > best_reward:
                            best_reward = total_reward
                            best_params = (hidden_layers, neurons, use_ln, use_dropout)
                            print(f"New best parameters: Hidden Layers: {best_params[0]}, Neurons: {best_params[1]}, Layer Norm: {best_params[2]}, Dropout: {best_params[3]} with reward: {best_reward}")
                            env.save_csv(f"outputs/{name}/ql-2x2_run{run}", episode)
                            shared_q_net.save(name)
                        else:
                            print(f"Hidden Layers: {hidden_layers}, Neurons: {neurons}, Layer Norm: {use_ln}, Dropout: {use_dropout} with reward: {total_reward}")

    for agent_id, agent in ql_agents.items():
        plt.plot(agent.rewards, label=f'Agent {agent_id}')
    end_time = time.time()
    # Calculate the total execution time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time} seconds")
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    plt.legend()
    # Save the plot before showing it
    plt.savefig('reward_best_params_2x2_pressure.png')
    plt.show()
    
    env.close()
