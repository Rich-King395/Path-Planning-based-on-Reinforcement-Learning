# Path Planning based on RL Algorithms
Implementing Reinforcement Learning (RL) Algorithms for global path planning in tasks of mobile robot navigation. Comparison analysis of Deep Q Network, Q-learning and Sarsa algorithms for the environment with obstacles.
## Description
RL Algorithms implemented in Python for the task of global path planning for mobile robot. Such system is said to have feedback. The agent acts on the environment, and the environment acts on the agent. At each step the agent:
* Executes action.
* Receives observation (new state).
* Receives reward.

The environment:
* Receives action.
* Emits observation (new state).
* Emits reward.

The following image shows the environment I estabilshed for this problem:
<img src="Figures/Map.png" alt="env" > 

Red square: starting position and target position

Blue blocks: obstacles

For SARSA and DQN, the state of the agent is the coordinate of it in the map. For DQN, the state of the agent is the image of the map at that time. The action space of the three algorithms are the same shown in the following image:
<img src="Figures/action.png" alt="env" width=400 height=400 > 

## Report
The final report of my final year project is available on arXiv:
```latex
@misc{ruiqi2024research,
      title={Research on Robot Path Planning Based on Reinforcement Learning}, 
      author={Wang Ruiqi},
      year={2024},
      eprint={2404.14077},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```


