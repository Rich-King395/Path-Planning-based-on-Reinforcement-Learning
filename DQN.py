import matplotlib.pyplot as plt

import random
import numpy as np
from Agent import DQNAgent
from env import Environment

TARGET_UPDATE = 5
num_episodes = 200
hidden = 128
gamma = 0.99
replay_buffer_size = 100000
batch_size = 128
eps_stop = 0.1
epsilon=eps = 0.6
Start_epsilon_decaying = 0
#End_epsilon_decaying = num_episodes // 1
End_epsilon_decaying = 200
epsilon_decaying = epsilon / (End_epsilon_decaying - Start_epsilon_decaying)

n_actions = 4
state_space_dim = 2
starting_position = [10, 10]
target_position=[90,90]
env = Environment( starting_position,target_position, 100, 100, n_actions)

if __name__=="__main__":
    agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
                 hidden, gamma)
    random.seed(20)
    env.reset()


    # Training loop
    cumulative_rewards = []
    Num_steps = []
    counter_reach_goal = 0


    for ep in range(num_episodes):
        # Initialize the environment and state
        #print('training started ...')
        state = env.reset()
        done = False
        eps -= epsilon_decaying
        epsilon = max(0.01, eps)
        cum_reward = 0
        counter = 0
        number_of_steps_taken_to_terminal = 0
        visited_X = []
        visited_Y = []
        #print("episode number: ",ep)
        while not done and counter < env.max_episode_steps:
            # if ep % 100 == 0:
            # env.render()
            # Select and perform an action
            action = agent.get_action(state, epsilon)

            visited_X.append(env.vector_agentState[0])
            visited_Y.append(env.vector_agentState[1])

            next_state,next_state_flag, reward, done, _ = env.step(action)
            #if counter%10 ==0:
            #print(next_state)

            cum_reward += reward
            
            agent.store_transition(state, action, next_state, reward, done) #储存经验
            agent.update_network() #更新策略网络参数

            state = next_state
            counter +=1
            number_of_steps_taken_to_terminal  += 1
        
        #print(state)
        if done:
          #print(state)
          print('number of steps taken by the agent: ', number_of_steps_taken_to_terminal)
          Num_steps.append(number_of_steps_taken_to_terminal)
          ###print(env.agentState[-5:])
          ###print(env.Collected_Data)
          
          ###if env.doneType != 0:
            ###print("Type of Terminal done flag: " ,env.doneType)
          cumulative_rewards.append(cum_reward)
          #if ep >= 50:
          #    last50_rewards.append(np.mean(cumulative_rewards[ep - 50:ep]))
          #    last50_steps.append(np.mean(Num_steps[ep -50: ep]))
          ###print("episode: %d: reward: %6.2f, epsilon: %.2f" % ( ep, cum_reward, epsilon))
          print("episode: %d: reward: %6.2f" % ( ep, cum_reward))
          print("**********************************************")

        # Update the target network, copying all weights and biases in DQN
        if ep % TARGET_UPDATE == 0:
            agent.update_target_network() #更新目标网络参数

    plt.figure(tight_layout=True)
    plt.plot(range(num_episodes), cumulative_rewards, label='cumulative rewards', color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Accumulated reward', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    #plt.savefig('R_3.eps',format = 'eps')
    plt.show()

    plt.figure(tight_layout=True)
    plt.plot(range(num_episodes), Num_steps, color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Taken steps', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('S_3.eps',format = 'eps',dpi=1200)
    plt.show()   

    ### Plot the trajectory
    x = np.append(np.array(visited_X), env.Terminal[0])
    y = np.append(np.array(visited_Y), env.Terminal[1])
    # x_s = np.array([50, 20, 80, 60, 50 ])
    # y_s = np.array([10, 60, 40, 60, 90])

    x_o = env.Obstacle_x 
    y_o = env.Obstacle_y

    plt.figure()

    #绘制矢量场图
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)

    #plt.scatter(x_s, y_s, c = 'k' ,marker = "o",label = 'Sensor')

    for i in range(len(x_o)):
        rectangle = plt.Rectangle(( 10* (x_o[i]-0.5), 10*(10 - y_o[i] -0.5)), 10, 10, fc='blue',ec="blue")
        plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Start")
    plt.scatter(90,90, marker = "s", ec = 'k', c ='red', s =50,label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('T_3.eps',format = 'eps')
    plt.show()
    ####################################





