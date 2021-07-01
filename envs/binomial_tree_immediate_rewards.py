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
            
        # Decode the action     
        pi_t = decode_action(action, self.actions)
        
        # Take one step according to binomial tree dynamics
        next_V_t = self.V_t * ( pi_t * self.dt * (np.random.choice(a=self.returns, size=1, replace=False, p=self.probs)[0] - self.r) + (1 + self.dt*self.r) )  # Update Wealth (see notes)
        
        # Update time-step
        self.time_state += 1
        
        done = self.time_state == self.num_timesteps           # Episode is finished if termination time is reached
        
        
        # Immediate Reward (U(V_t+1) - U(V_t))                                 
        if self.utility == "log":                              # Logarithmic utility function
            reward = np.log(next_V_t) - np.log(self.V_t)
        elif self.utility == "sqrt":                           # Square root utility function
            reward = np.sqrt(next_V_t) - np.sqrt(self.V_t)
        else:
            raise ValueError("Utility function {} not implemented.".format(self.utility))
            
            
        # Update wealth state
        self.V_t = next_V_t
            
        return self._get_obs(), reward, done, {}          # {} empty info
    
    def _get_obs(self):
        '''Get observation from environment'''
        return (self.time_state, self.V_t)
            
    def reset(self):
        '''Reset the state of the environment to an initial state'''
        self.time_state   = 0                                          # setting time to zero
        self.V_t          = self.V_0                                   # setting wealth to V_0
        return self._get_obs()