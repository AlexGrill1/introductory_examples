import gym
import numpy as np
from gym import spaces
import pandas as pd
import math


def encode_wealth(wealth, wealth_bins):
    return(pd.cut(x=[wealth], bins=wealth_bins, right=False, retbins=True)[0][0])
    
def decode_wealth(discr_wealth, wealth_bins):
    return [wealth_bins[discr_wealth], wealth_bins[discr_wealth + 1]]
    
def encode_action(action, actions):
    return(int(np.where(action == actions)[0][0]))

def decode_action(action, actions):
    return(actions[action])
    

class BSEnv(gym.Env):
    '''Custom discrete-time Black-Scholes environment with one risky-asset and bank account'''
    metadata = {'render.modes': ['human']}
    
    def __init__(self, mu, sigma, r, T, dt, V_0, actions, wealth_bins, U_2=np.log, batch_size=1):
        '''
        Args:
            :params mu (float):         expected risky asset return
            :params sigma (float):      risky asset standard deviation
            :params r (float):          risk-less rate of return
            :params T (float):          investment horizon
            :params dt (float):         time-step size
            :params V_0 (float, tuple): initial wealth, if tuple (v_d, v_u) draws initial wealth V(0) uniformly from [v_d, v_u]
            :params actions (np.array): possible investment fractions into risky asset
            :params wealth_bins (np.array): contains the limits of each wealth bin in ascending order
            :params U_2 (callable):     utility function for terminal wealth
            :params batch_size (int):   specifies how many samples are drawn for one step, i.e. batch_size=2 invests 2 times for one period. 
                                        Returns batch_size many next_states and rewards
        '''
        
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
        self.U_2 = U_2
        self.batch_size = int(batch_size)
        
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
        
        # Decode the discrete action to a float (investment choice)
        pi_t = decode_action(action, self.actions)
        
        # Update Wealth (see wealth dynamicy, Inv. Strategies script (by Prof. Zagst) Theorem 2.18):
        # 1) Sample BM batch_size many increments for one step
        dW_t = np.random.normal(loc=0, scale=math.sqrt(self.dt), size=self.batch_size)
        # 2) Wealth process update via simulation of the exponent
        self.next_V_t = self.V_t * np.exp( (self.r + pi_t*(self.mu - self.r) - 0.5*(pi_t**2)*(self.sigma**2)) * self.dt + pi_t*self.sigma*dW_t)  #size batch_size
        self.V_t = self.next_V_t[0]
        
        
        # Old
        #self.V_t *= pi_t * (np.random.normal(loc=self.dt * self.mu, scale=math.sqrt(self.dt) * self.sigma) - self.dt*self.r) + (1 + self.dt*self.r)  # Update Wealth (see notes)
        #self.V_t *= pi_t * self.dt * (self.mu  - self.r) + (1 + self.dt * self.r)    # stock pays deterministic higher return
        
        
        self.time_state += 1      # updating time-step
        #self.wealth_state = encode_wealth(self.V_t, self.wealth_bins)
        
        done = self.time_state == self.num_timesteps           # Episode is finished if termination time is reached
        
        reward = np.zeros(self.batch_size)                     # Reward is zero for each time step t<T
        if done:                                          # Reward at termination time R_T = U(V_T)
            reward = self.U_2(self.next_V_t)
            
        return self._get_obs(self.next_V_t), reward, done, {}          # {} empty info
    
    
    def _get_obs(self, wealth):
        '''Returns a np.array of length batch_size containing tuples (t,v) of states'''
        
        
        return np.array([(self.time_state*self.dt, encode_wealth(v, self.wealth_bins)) for v in wealth])
            
    
    def reset(self):
        '''Reset the state of the environment to an initial state'''
        
        self.time_state   = 0                                          # setting time to zero
        
        if isinstance(self.V_0, tuple):
            # Draws uniform from the interval specified in self.V_0
            self.V_t = np.random.uniform(low=self.V_0[0], high=self.V_0[1], size=1)
        else:    
            self.V_t = np.array([self.V_0], dtype="float64")      # setting wealth to V_0
        
        return self._get_obs(self.V_t)
    
    
    #def render(self, mode='human', close=False):
    # Render the environment to the screen
    #...
    
    
    
    
    
def transform_Q_interval_to_Q_numeric(Q_int):
    '''Changes the keys of the Q-Table from interval representation (t, interval) to numeric representation
       (t, mid.interval) by choosing the middle point of the interval.
    
    Args:
    :params Q_int [dict]: dictionary with Q-Values
    
    Returns:
    - Q_numeric [dict]: A dictionary containing the Q-Table with states in numeric representation.
    '''

    # initialise the new Q-Table
    Q_numeric = dict()

    for key, value in Q.items():
        Q_numeric[(key[0], key[1].mid)] = value
    
    return Q_numeric


def transform_Q_numeric_to_Q_interval(Q_numeric, wealth_bins):
    '''Changes the keys of the Q-Table from numeric representation (t, mid.interval) to interval representation
       (t, interval) given by wealth_bins.
    
    Args:
    :params Q_numeric [dict]: dictionary with Q-Values
    :params wealth_bins [np.array]: array containint interval limits
    
    Returns:
        - Q_interval [dict]: A dictionary containing the Q-Table with states in interval representation.
    '''
    
    # initialise the new Q-Table
    Q_interval = dict()
    
    for key, value in Q_numeric.items():
        Q_interval[(key[0], encode_wealth(key[1], wealth_bins))] = value
        
    return Q_interval


def state_to_numeric(state):
    '''Transforms the interval representation of a state (t, interval) to a numeric representation (t, mid.intverval)
    
    Args:
    :params state [Tuple(int, pd.Interval)]: A state representing the timestep and wealth as interval
    
    Returns a tuple (t, mid.interval) if interval is not the lowest [0, lower) or highest interval [upper, +inf)
    '''
    if state[1].right == float('inf'):
        # return the lower bound of the highest interval [upper, +inf)
        return (state[0], state[1].left)
    
    elif state[1].left == 0:
        # return the upper bound of the lowest interval [0, upper)
        return (state[0], state[1].right)
    
    else:
        # return the mid of the interval
        return((state[0], state[1].mid))