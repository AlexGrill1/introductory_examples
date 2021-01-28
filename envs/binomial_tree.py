import gym
import numpy as np
from gym import spaces
import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import random


def encode_wealth(wealth, wealth_bins):
    '''Encodes the wealth from [0, +Inf) into the corresponding wealth_bin and returns the wealth_bin encoded as integer'''
    return(pd.cut(x=[wealth], bins=wealth_bins, right=False, labels=False)[0])
    
def encode_action(action, actions):
    '''Encodes an action as the corresponding index in actions'''
    return(int(np.where(action == actions)[0][0]))

def decode_action(action, actions):
    '''Decodes the index as the corresponding investment in the risky asset'''
    return(actions[action])
    

class BinomialTree(gym.Env):
    '''Custom binomial stock price tree environment with one risky-asset and bank account'''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, up_prob, up_ret, down_ret, r, T, dt, V_0, actions, utility):
        assert divmod(T, dt)[1] == 0        # To-Do: change to ValueError, is T 'ganzzahlig' divisible
        super().__init__()
        
        self.probs      = np.array([up_prob, 1-up_prob])    # probabilities for an upward/downward step of the risky asset
        self.returns    = np.array([up_ret, down_ret])      # risky asset return for an upward/downward step
        self.r = r                                          # risk-free rate (bank account return, riskless)
        self.T = T                                          # Termination time
        self.dt = dt                                        # time-step size
        self.num_timesteps = T//dt
        self.V_0 = V_0                                      # Initial wealth
        self.actions = actions                              # possible actions, fraction of wealth invested in risky aset
        self.num_actions = len(self.actions)                # number of possible actions
        #self.wealth_bins = wealth_bins                      # discrete wealth space
        self.utility = utility                              # "log" or "sqrt"
        
        self.reset()                                        # Set environment to initial state (0, V_0)
        
        # Action space
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Observation space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.num_timesteps),
            spaces.Box(low=np.array([0]), high=np.array([float("inf")])) ))
        #self.num_observation_space = (self.num_timesteps + 1) * (len(wealth_bins) - 1)
        
    
    def step(self, action):
        '''Execute one time step within the environment'''
        
        """
        Samples at each time step from the vector of returns [up_ret, down_ret] with probabilities p=[up_prob, 1-up_prob], and calculates
        the new wealth according to the investment in the riskless asset and the risky asset. 
        """
        
        assert self.action_space.contains(action)
            
        pi_t = decode_action(action, self.actions)
        self.V_t *= pi_t * self.dt * (np.random.choice(a=self.returns, size=1, replace=False, p=self.probs)[0] - self.r) + (1 + self.dt*self.r)  # Update Wealth (see notes)
        self.time_state += 1      # updating time-step
        
        done = self.time_state == self.num_timesteps           # Episode is finished if termination time is reached
        
        reward = 0                                             # Reward is zero for each time step t<T
        if done:                                               # Reward at termination time R_T = U(V_T)
            if self.utility == "log":                              # Logarithmic utility function
                reward = np.log(self.V_t)
            elif self.utility == "sqrt":                           # Square root utility function
                reward = np.sqrt(self.V_t)
            else:
                raise ValueError("Utility function {} not implemented.".format(self.utility))
            
        return self._get_obs(), reward, done, {}          # {} empty info
    
    def _get_obs(self):
        '''Get observation from environment'''
        return (self.time_state, self.V_t)
            
    def reset(self):
        '''Reset the state of the environment to an initial state'''
        self.time_state   = 0                                          # setting time to zero
        self.V_t          = self.V_0                                   # setting wealth to V_0
        return self._get_obs()
    
    
    #def render(self, mode='human', close=False):
    # Render the environment to the screen
    #...
    

def fun(x, y, Q, actions):
    '''Help function used in plot_q_values'''
    # Returns the Q-values for each state-action pair at time step 1.
    return np.array([Q[(1,wealth)][encode_action(action, actions)] for action, wealth in zip(x,y)])
    
def plot_q_values(Q, actions):
    '''Creates a 3d Wireframe plot of the Q-value function for each state-action pair and adds the predicted action (i.e. argmax_a Q(s,a)'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = actions
    y = np.array(sorted([wealth for t, wealth in Q.keys() if t == 1]))
    X, Y = np.meshgrid(x, y)
    zs = np.array(fun(np.ravel(X), np.ravel(Y), Q, actions))
    Z = zs.reshape(X.shape)

    # Predicted Actions for each state
    states = [key for key in Q.keys() if key[0] == 1]
    predicted_actions = [decode_action(np.argmax(Q[state]), actions) for state in states]
    wealths = [wealth for _, wealth in states]
    predicted_Q_values = [Q[state][np.argmax(Q[state])] for state in states]

    ax.plot_wireframe(X, Y, Z, color="black")

    ax.set_xlabel('investment in risky asset')
    ax.set_ylabel('wealth')
    ax.set_zlabel('Q-values')
    ax.scatter(predicted_actions, wealths, predicted_Q_values, zdir="z", c="red", alpha=1, label="Predicted Actions")
    plt.title("Learned Q-value surface")
    ax.legend()

    plt.show()