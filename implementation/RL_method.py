"""
Created on Tue Jan  5 15:11:58 2018
@author: eric.benhamou, david.sabbagh
"""

#import declaration    
import numpy as np

"""
Reinforcement method:
support Q Learning and SARSA
"""
class Dynamic_Programming:
    def __init__(self, dim_states, dim_actions, algorithm='QLearning', learn_type='TD',learning_rate=0.1, discount_rate=0.9 ):
        #some validations
        if algorithm != 'Q-Learning' and algorithm != 'SARSA' : 
            raise ValueError('invalid algorithm: supported Q-Learning and SARSA')
        self.algorithm = algorithm
        if learn_type != 'TD' and learn_type != 'MC' : 
            raise ValueError('invalid learn_type: supported MC and TD')
        self.learn_type = learn_type

        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.learning_rate = learning_rate
        self.discount_rate =  discount_rate
        self.Q = np.zeros((self.dim_states,self.dim_actions))

    ''' simple method to initialize the Q matrix
    '''
    def initialize(self,mean=0.0,sigma=0.01):
        self.Q = np.random.normal(mean,sigma,(self.dim_states,self.dim_actions))


    ''' best action for a given state
    '''
    def best_action(self,state):
        return np.argmax(self.Q[state,:])


    '''
    Update Q value table from transition data:
        inputs: s, a are states and action index 
        reward is the current reward, 
        terminal is a 0 or 1 variable to say if we
        are in the terminal stage, next_s, next_a are the next state and action
        index
    '''
    def learn(self,s,a,reward,terminal,next_s,next_a=-1):
        if self.algorithm == 'SARSA' and next_a < 0 : 
            raise ValueError('invalid na')
        # compute the increment
        if self.learn_type == 'TD': 
            if self.algorithm == 'Q-Learning':
                increment = reward+self.discount_rate*(1.-terminal)*np.max(self.Q[next_s,:])-self.Q[s,a]
            elif self.algorithm == 'SARSA' :
                increment = reward+self.discount_rate*(1.-terminal)*self.Q[next_s,next_a]-self.Q[s,a]
        elif self.learn_type == 'MC':
            increment = reward-self.Q[s,a]
        # add increment
        self.Q[s,a] += self.learning_rate*increment
          
    
    '''
    calculate episode return (for MC learning case)
    inputs :
    rewards should be a numpy array of rewards for one episode
    outputs : episode return (numpy array of shape n)
    '''
    def compute_episode_return(self,rewards):
        # make a full copy of rewards to avoid side effect
        episode_return = np.array(rewards.copy())
        for i in range(int(episode_return.shape[0]-2),-1,-1): 
            episode_return[i]+=self.discount_rate*episode_return [i+1]
        return episode_return 
        
        
        
