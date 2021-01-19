import gym
import numpy as np
from gym import spaces
import pandas as pd
import math


def encode_wealth(wealth, wealth_bins):
        return(pd.cut(x=[wealth], bins=wealth_bins, right=False, labels=False)[0])
    
def encode_action(action, actions):
        return(int(np.where(action == actions)[0][0]))

def decode_action(action, actions):
        return(actions[action])
    

class BSEnv(gym.Env):
    '''Custom discrete-time Black-Scholes environment with one risky-asset and bank account'''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, mu, sigma, r, T, dt, V_0, actions, wealth_bins):
        assert divmod(T, dt)[1] == 0        # To-Do: change to ValueError, is T 'ganzzahlig' divisible
        super().__init__()
        self.mu    = mu                        # risky asset return
        self.sigma = sigma                     # risky asset volatility
        self.r = r                             # risk-free rate (bank account return, riskless)
        self.T = T                             # Termination time
        self.dt = dt                           # time-step size
        self.num_timesteps = T//dt
        self.V_0 = V_0                         # Initial wealth
        self.actions = actions                 # possible actions, fraction of wealth invested in risky aset
        self.num_actions = len(self.actions)   # number of possible actions
        self.wealth_bins = wealth_bins
        
        self.reset()
        
        # Action space
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Observation space
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.num_timesteps),
            spaces.Discrete(len(self.wealth_bins))))
        
    
    def step(self, action):
        '''Execute one time step within the environment'''
        assert self.action_space.contains(action)
        
        pi_t = decode_action(action, self.actions)
        #self.V_t *= pi_t * (np.random.normal(loc=self.dt * self.mu, scale=math.sqrt(self.dt) * self.sigma) - self.dt*self.r) + (1 + self.dt*self.r)  # Update Wealth (see notes)
        self.V_t *= pi_t * self.dt * (self.mu  - self.r) + (1 + self.dt * self.r)    # stock pays deterministic higher return
        self.time_state += 1      # updating time-step
        #self.wealth_state = encode_wealth(self.V_t, self.wealth_bins)
        
        done = self.time_state == self.num_timesteps           # Episode is finished if termination time is reached
        
        reward = 0                                        # Reward is zero for each time step t<T
        if done:                                          # Reward at termination time R_T = U(V_T)
            reward = np.log(self.V_t)
            
        return self._get_obs(), reward, done, {}          # {} empty info
    
    def _get_obs(self):
        return (self.time_state, encode_wealth(self.V_t, self.wealth_bins))
            
    def reset(self):
        '''Reset the state of the environment to an initial state'''
        self.time_state   = 0                                          # setting time to zero
        self.V_t          = self.V_0                                   # setting wealth to V_0
        #self.wealth_state = encode_wealth(self.V_t, self.wealth_bins)  # encoding wealth V_0 
        return self._get_obs()
    
    
    #def render(self, mode='human', close=False):
    # Render the environment to the screen
    #...