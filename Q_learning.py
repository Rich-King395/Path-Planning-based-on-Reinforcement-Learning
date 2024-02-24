from env import Environment
from env import final_states 
from env import obstacle_width
from Agent import QLearningTable
import matplotlib.pyplot as plt
import numpy as np

gamma = 0.99
num_episodes=500
epsilon=eps =0.2
Start_epsilon_decaying = 0
#End_epsilon_decaying = num_episodes // 1
End_epsilon_decaying = num_episodes
epsilon_decaying = epsilon / (End_epsilon_decaying - Start_epsilon_decaying)

n_actions = 8
starting_position = [10, 0]
target_position=[90,100]
env = Environment( starting_position,target_position, 100, 100, n_actions)

def update():
    # Resulted list for the plotting Episodes via Steps
    Num_steps = []
    # Summed costs for all episodes in resulted list
    cumulative_rewards = []

    final_path=[]
    visited_X = [starting_position[0]]
    visited_Y = [starting_position[1]]

    for ep in range(num_episodes): #200个回合
        # Initial state
        state = env.reset() #智能体回到起点并清空d字典
        done = False
        global eps
        eps -= epsilon_decaying
        epsilon = max(0.01, eps)
        cum_reward = 0 # Cummulative reward  for each episode
        number_of_steps_taken_to_terminal = 0 # Updating number of Steps for each Episode
        visited_X_final = []
        visited_Y_final = []
        
        while not done :
            # agent chooses action based on state当前机器人的坐标位置
            action = agent.get_action(str(state),epsilon) #寻找动作的依据为以一定概率选择目前状态下动作值函数最大的动作，以一定概率随机选择（随机选择的目的是增加探索率）

            visited_X_final.append(env.vector_agentState[0])
            visited_Y_final.append(env.vector_agentState[1])

            # agent takes an action and get the next state and reward
            next_state, next_state_flag,reward, done, _ = env.step(action) #将该动作执行，得到奖励值，下个状态以及是否结束寻路标志

            # agent learns from this transition and calculating the cost
            cum_reward += agent.learn(str(state), action, reward, str(next_state),next_state_flag)

            # Swapping the states - current and next
            state = next_state

            # Calculating number of Steps in the current Episode
            number_of_steps_taken_to_terminal += 1

            # Break while loop when it is the end of current Episode
            # When agent reached the goal or obstacle
            if done:
                print('number of steps taken by the agent: ', number_of_steps_taken_to_terminal)
                Num_steps.append(number_of_steps_taken_to_terminal)
                cumulative_rewards.append(cum_reward)
                #if ep >= 50:
                #    last50_rewards.append(np.mean(cumulative_rewards[ep - 50:ep]))
                #    last50_steps.append(np.mean(Num_steps[ep -50: ep]))
                ###print("episode: %d: reward: %6.2f, epsilon: %.2f" % ( ep, cum_reward, epsilon))
                print("episode: %d: reward: %6.2f" % ( ep, cum_reward))
                print("**********************************************")
                break

    # Showing the Q-table with values for each action
    agent.print_q_table()
    # Showing the final route
    env.final()

    plt.figure(tight_layout=True)
    plt.plot(range(num_episodes), cumulative_rewards, label='cumulative rewards', color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Accumulated reward', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('Q_learning_Accumulated_Reward.eps',format = 'eps')

    plt.figure(tight_layout=True)
    plt.plot(range(num_episodes), Num_steps, color='b')
    plt.xlabel('Episode',size = '14')
    plt.ylabel('Taken steps', size = '14')
    plt.grid(False)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.savefig('Q_learning_Steps_per_Episode.eps',format = 'eps',dpi=1200)

    ### Plot the trajectory
    final_path=list(final_states().values())
    print(final_path)
    for i in range(len(final_path)):
        visited_X.append(final_path[i][0])
        visited_Y.append(final_path[i][1])

    ### Plot the trajectory
    x_shortest = np.append(np.array(visited_X), env.Terminal[0])
    y_shortest = np.append(np.array(visited_Y), env.Terminal[1])

    x_final = np.append(np.array(visited_X_final), env.Terminal[0])
    y_final = np.append(np.array(visited_Y_final), env.Terminal[1])
     
    # x_s = np.array([50, 20, 80, 60, 50 ])
    # y_s = np.array([10, 60, 40, 60, 90])

    x_o = env.Obstacle_x 
    y_o = env.Obstacle_y

    plt.figure()
    #绘制矢量场图
    plt.quiver(x_shortest[:-1], y_shortest[:-1], x_shortest[1:]-x_shortest[:-1], y_shortest[1:]-y_shortest[:-1], scale_units='xy', angles='xy', scale=1)

    #plt.scatter(x_s, y_s, c = 'k' ,marker = "o",label = 'Sensor')

    for i in range(len(x_o)):
        rectangle = plt.Rectangle(( 10* (x_o[i]-0.5), 10*(10 - y_o[i] -0.5)), obstacle_width, obstacle_width, fc='blue',ec="blue")
        plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(starting_position[0],starting_position[1], marker = "s", ec = 'k', c ='red', s=100, label ="Start")
    plt.scatter(target_position[0],target_position[1], marker = "s", ec = 'k', c ='red', s =100,label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Q_learning_Shortest_Path.eps',format = 'eps')

    plt.figure()
    #绘制矢量场图
    plt.quiver(x_final[:-1], y_final[:-1], x_final[1:]-x_final[:-1], y_final[1:]-y_final[:-1], scale_units='xy', angles='xy', scale=1)

    #plt.scatter(x_s, y_s, c = 'k' ,marker = "o",label = 'Sensor')

    for i in range(len(x_o)):
        rectangle = plt.Rectangle(( 10* (x_o[i]-0.5), 10*(10 - y_o[i] -0.5)), obstacle_width, obstacle_width, fc='blue',ec="blue")
        plt.gca().add_patch(rectangle)

    #plt.scatter(10,10, marker = "s", ec = 'k', c ='red', s=50, label ="Terminal")
    plt.scatter(starting_position[0],starting_position[1], marker = "s", ec = 'k', c ='red', s=100, label ="Start")
    plt.scatter(target_position[0],target_position[1], marker = "s", ec = 'k', c ='red', s =100,label="Target")
    plt.grid(linestyle=':')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.xlabel('x (m)',size = '14')
    plt.ylabel('y (m)',size = '14')
    #plt.legend(loc=4)
    plt.xticks(size = '12')
    plt.yticks(size = '12')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('Q_learning_Final_Path.eps',format = 'eps')
    plt.show()
    # # Plotting the results
    # agent.plot_results(steps, all_costs)


if __name__ == "__main__":
    agent = QLearningTable(actions=list(range(n_actions)),
                    learning_rate=0.1,
                    reward_decay=0.9,
                    e_greedy=0.9) #初始化
    
    update() #学习过程

    

