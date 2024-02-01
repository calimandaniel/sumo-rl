# Multi-Agent Reinforcement Learning for Traffic Signal Control
Caliman Ștefan-Daniel - 2122749
Vilgot Astrom - 2115562


## Overview

This is a multi-agent DQN solution for a traffic signal control scenario. It is implemented on top of [SUMO-RL](https://github.com/LucasAlegre/sumo-rl.git), a reinforcement learning interface for [SUMO](https://github.com/eclipse/sumo) which is a traffic simulation environment commonly used in research and industry. The implemention is based on [this paper](https://doi.org/10.48550/arXiv.2204.12190), although a lot was changed and focus shifted as the scope of this project differs. The paper is about deciding how much information to share between agents, and we decided to focus on three scenarios:
- No communitcation - Each agent acts alone, only aware of its own existance.
- Shared Q-network - Each agent only considers its own observations, but they all share the same Q-network and will affect each other's actions.
- Local observation sharing - Each agent considers its own observation as well as the observations of its direct neighbors.

## Table of Contents

- [Usage](#usage)
- [Observation Space](#observation-space)
- [Rewards](#rewards)
- [Metrics](#metrics)
- [Action Space](#action-space)
- [Training](#training)

## Usage
### Installing SUMO
Clone the SUMO repository for latest features with
```bash
git clone --recursive https://github.com/eclipse-sumo/sumo
```

Alternitvely, it can be downloaded at [their site](https://sumo.dlr.de/docs/Downloads.html). See the [SUMO repo](https://github.com/eclipse/sumo) for more info on this.

### Installing SUMO-RL
Simple installiation with pip:
```bash
pip install sumo-rl
```

### Running our trained models
Our trained models can be found in the models folder. To run a model and watch it perform, you can use the file load_and_gui.py with the model name as the -n argument. For example, to run the shared Q-network model using the queue reward function, simply run:
```bash
python .\experiments\load_and_gui.py -n .\models\good_queue_reward_big_sharedQ
```
It is also possible to run an evaluation loop for plotting results:
```bash
python .\experiments\evaluate.py -n "good_queue_reward_big_sharedQ"
```
This will create a csv file with information about the environment at every 5th step, which can be used for plotting. It can be found in a folder "good_queue_reward_big_sharedQ" inside an eval folder in the output folder.

To visualize results, the file plot.py can be used:
```bash
python .\outputs\plot.py -f .\outputs\evals\good_queue_reward_big_sharedQ\
```
Use the argument -func for different metrics:
```python
Metric to plot:
1: waiting time,
2: number of vehicles,
3: number of stopped vehicles vs total number of vehicles,
4: Mean speed,
5: agents total accumulated waiting time,
6: Mean Speed / Number of Cars,
7: Mean Waiting Time.
```

### Training new model
New models are trained with the test_experiment.py file. 
At the top of the file, the following hyper-parameters can be set:
```python
alpha = 0.01
gamma = 0.99
decay = 0.9999
runs = 2
episodes = 2

name = "good_queue_reward_big_sharedQ"
```

When initializing a new SumoEnvironment, the following hyper.parameters can be set:
```python
net_file="nets/2x2grid/2x2.net.xml",
route_file="nets/2x2grid/2x2.rou.xml",
use_gui=False,
num_seconds=50000,
min_green=5,
delta_time=5,
reward_fn="queue"
```

The exact values used here are example values we used when training the good_queue_reward_big_sharedQ.pt model. 

### Local Observation Sharing
When implementing local observation sharing, changes had to be made to dependent files. So, to train a model using communication of observations between neighbors, use the postfix _com when running files. For example:
```bash
python .\experiments\test_experiment_com.py
```


## Observation Space

At every time step, each agent can observe its current action (phase), whether changing action is possible, density of the incoming lanes, and queue in the incoming lanes. 

Density is defined as the number of vahicles in a lane divided by the maximum possible number of vehicles in that lane.

Queue is defined as the number of halting vehicles in a lane divided by the maximum possible number of vehicles in that lane.

### Observation Sharing
For the tests using local observation sharing, neighboring agents share their full observations with eachother. This means that if an agent has, for example, two neighbors, it will consider a state of the size three times its own observation space.

## Rewards

The primary reward used is the queue reward, which is defined as the number of halting veichles in every lane connected to the intersection, multiplied by -1. We used this as the primary reward function because it is what is used in the [paper](https://doi.org/10.48550/arXiv.2204.12190). 

<img src="https://latex.codecogs.com/svg.image?\bg{white}&space;r=-\sum_{i=1}^{n}H_i&space;" title=" r=-\sum_{i=1}^{n}H_i " />


The secondary reward, used for evaluation of the primary reward, is the pressure reward, which is defined as the number of vehicles on outgoing lanes subtracted by the number of vehicles on incoming lanes.

<img src="https://latex.codecogs.com/svg.image?\bg{white}&space;r=\sum_{i=1}^{n}V_i-\sum_{j=1}^{m}V_j&space;" title=" r=\sum_{i=1}^{n}V_i-\sum_{j=1}^{m}V_j " />


## Action Space
The action space is discrete, with 4 possible actions for each agent. As described in the [SUMO-RL](https://github.com/LucasAlegre/sumo-rl.git) repo:

<p align="center">
<img src="./docs/_static/actions.png" align="center" width="75%"/>
</p>

Actions can only be updated when a minimum green time and a yellow time has passed, which agents are made aware of in the observation.


