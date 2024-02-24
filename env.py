import numpy as np
import warnings

thr = 10 # threshold distance to the terminal for making decision of the done flag
v = 10 #机器人每次移动距离

# warnings.simplefilter("error")
warnings.simplefilter("ignore", UserWarning)

# Global variable for dictionary with coordinates for the final route
final_route = {}

class Environment(object):
  def __init__(self, initial_position, target_position,X_max, Y_max, num_actions):
    #Initial state of the system:
    self.state0 = np.zeros((2,11,11)) #初始状态,三位数组,维度为3*11*11，所有元素都初始化为0
    self.state0[0][9][1] = 1 # robot initial position

    self.Obstacle_x = [0 , 1, 1, 1, 1, 2 ,4 ,4 ,4 ,4 ,5, 5, 5, 6, 8,8 ,8 , 9, 9, 9,9, 10, 10]
    self.Obstacle_y = [2, 6, 5, 2,1,1, 8,7, 4, 3, 7,6 , 3, 6, 10, 9, 3, 10, 9, 4, 3, 10, 9]

    self.vector_obstacle_x=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    self.vector_obstacle_y=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(self.Obstacle_x)):
      self.vector_obstacle_x[i]=10*(self.Obstacle_x[i]-0.5)
      self.vector_obstacle_y[i]=10*(10 - self.Obstacle_y[i] -0.5)
    
    self.obstacle =  [np.zeros((1, 4)).tolist() for i in range(23)]
    for i in range(len(self.vector_obstacle_x)):
      self.obstacle[i]=[self.vector_obstacle_x[i],self.vector_obstacle_y[i],10,10]

    #将self.Obstacle_x和self.Obstacle_y中的元素作为索引，在self.state0数组中的特定位置赋值为1
    for i in range(len(self.Obstacle_x)):
      self.state0[1, self.Obstacle_y[i], self.Obstacle_x[i]] = 1 #将障碍物位置的元素值初始化为1

    self.state0[1][1][9] = 1 #the position of the Terminal
    self.X_max = X_max #range of X: X_max, the min is 0，X轴范围
    self.Y_max = Y_max #range of Y: Y_max, the min is 0，Y轴范围
    self.vector_state0 = np.asarray(initial_position) #initial state，robot initial position, (10,10)
    self.Is_Terminal = False #achieve terminal or not, bool value, terminal = start point
    self.vector_agentState = np.copy(self.vector_state0) # state of the agent
    self.agentState = np.copy(self.state0) # state of the agent
   # self.visited_charger = 0 #visit the charge or not

    self.Terminal = np.asarray(target_position) #np.asarray([90., 90.]) # terminal 2
    self.doneType = 0 # flag showing type of done! 是否完成寻路
    self.max_episode_steps = 5000 #每个回合最大步数
    self.steps_counter = 0 #步数计数器
    self.num_actions = num_actions #number of actions

  # Dictionaries to draw the final route
    self.dic = {}
    self.final_path = {}
    # Key for the dictionaries
    self.index = 0
    # Writing the final dictionary first time
    self.firstsuc= True
    # Showing the steps for longest found route
    self.longest = 0
    # Showing the steps for the shortest route
    self.shortest = 0

    #self.actionspace = {0: [0,0], 1:[v,0], 2:[v,v], 3: [0,v], 4: [-v,v], \
    #                    5:[-v,0], 6:[-v,-v], 7:[0,-v], 8: [v,-v]}
    self.actionspace = {0:[v,0], 1:[0,v], 2: [-v,0], 3: [0,-v]} #action space
    
  def reset(self): #环境重置
    self.agentState = np.copy(self.state0)
    self.vector_agentState = np.copy(self.vector_state0)
    self.dic = {}
    self.index=0
    self.doneType = 0
    self.steps_counter = 0
    self.Is_Terminal = False
    return self.agentState

  def step(self, action): #agent interact with the environment through action
    V = self.actionspace[action]
    self.vector_agentState[0] += V[0] 
    self.vector_agentState[1] += V[1] 
    #if agent cross the boundary
    if self.vector_agentState[0] < 0:
      self.vector_agentState[0] = 0
    if self.vector_agentState[0] > 100:
      self.vector_agentState[0] = 100
    if self.vector_agentState[1] < 0:
      self.vector_agentState[1] = 0
    if self.vector_agentState[1] > 100:
      self.vector_agentState[1] = 100

    # Writing in the dictionary coordinates of found route
    self.dic[self.index] = self.vector_agentState #将坐标加入路径字典

    # Updating key for the dictionary
    self.index += 1

    i_x = np.copy(self.vector_agentState[0])/10 #x坐标变换
    i_y = 10 - np.copy(self.vector_agentState[1])/10 #x坐标变换
    self.agentState = np.copy(self.state0) #2*11*11
    self.agentState[0][9][1] = 0
    self.agentState[0, int(i_y), int(i_x)] = 1 #智能体移动后的位置
    #self.energy_level -= self.propulsion_power(V) * T_s
    self.steps_counter +=1 #step accumulate 1
    self.Is_Terminal = self.isTerminal() # achieve the terminal or not
    
    reward,next_state_flag = self.get_reward(self.vector_agentState)     

    return self.agentState, next_state_flag,reward, self.Is_Terminal , None

  # function for judging whether agent achieve the terminal or not
  def isTerminal(self):
    #计算目前状态和终点间的欧几里德距离
    Distance2Terminal = np.linalg.norm(np.subtract(self.vector_agentState , self.Terminal))
    #智能体到达终点，收集的数据满足要求，且能量没有用完，这完成任务
   # if d_.all() == True and Distance2Terminal**0.5 == 0 and self.energy_level > 0: ###self.agentState[2] > 0 :
    if Distance2Terminal**0.5 == 0: 
      self.doneType = 1
      return True
    else:
      return False

#function for geting rewards
  def get_reward(self,state):
#    ch, dist = self.channel()
    reward = 0 # initialize the reward as 0
    #Cooridinate change
    i_x = int(np.copy(self.vector_agentState[0])/10)
    i_y = int(10 - np.copy(self.vector_agentState[1])/10)
    
    # agent doesn't achieve the terminal
    if not self.Is_Terminal: 
       #judge whether the agent  crash the obstacle
      if self.is_collision(state):
          reward=-20
          next_state_flag = 'obstacle'
      else:
          reward=-1
          next_state_flag = 'continue'

    elif self.doneType == 1:
        reward = 20
        next_state_flag = 'goal'
        # Filling the dictionary first time，第一次找到完整路径
        if self.firstsuc == True:
            for j in range(len(self.dic)):
                self.final_path[j] = self.dic[j]
            self.firstsuc = False
            self.longest = len(self.dic)
            self.shortest = len(self.dic)
      # Checking if the currently found route is shorter
        if len(self.dic) < len(self.final_path):
            # Saving the number of steps for the shortest route
            self.shortest = len(self.dic)
            # Clearing the dictionary for the final route
            self.final_path = {}
            # Reassigning the dictionary
            for j in range(len(self.dic)):
                self.final_path[j] = self.dic[j] #将当前路径作为最终路径

        # Saving the number of steps for the longest route
        if len(self.dic) > self.longest:
            self.longest = len(self.dic)
    return reward, next_state_flag 
  
  # Function to show the found route
  def final(self):
      # Showing the number of steps
      print('The shortest route:', self.shortest)
      print('The longest route:', self.longest)
      for j in range(len(self.final_path)):
          #Showing the coordinates of the final route
          #print(self.final_path[j])
          final_route[j] = self.final_path[j]

  def is_collision(self,state):
    delta = 0.5
    for (x, y, w, h) in self.obstacle: #与矩形障碍物相撞
      if 0 <= state[0] - (x - delta) <= w + 2 * delta \
            and 0 <= state[1] - (y - delta) <= h + 2 * delta:
        return True

# Returning the final dictionary with route coordinates
# Then it will be used in agent.py
def final_states():
    return final_route
