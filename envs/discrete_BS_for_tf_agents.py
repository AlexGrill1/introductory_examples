import gym
import numpy as np
from gym import spaces
import pandas as pd
import math


    

class BSEnv(gym.Env):
    '''
    Custom discrete-time Black-Scholes environment with one risky-asset and bank account
    The environment simulates the evolution of the investor's portfolio according to a discrete version of the wealth SDE.
    
    At the end of the investment horizon the reward is equal to U(V(T)), else zero.
    '''
    
    metadata = {'render.modes': ['human']}
    
    
    def __init__(self, mu, sigma, r, T, dt, V_0 = 100, U_2 = math.log):
        '''
        :params mu (float):         expected risky asset return
        :params sigma (float):      risky asset standard deviation
        :params r (float):          risk-less rate of return
        :params T (float):          investment horizon
        :params dt (float):         time-step size
        :params V_0 (float, tuple): initial wealth, if tuple (v_d, v_u) draws initial wealth V(0) uniformly from [v_d, v_u]
        :params U_2 (callable):     utility function for terminal wealth
        '''
        
        assert divmod(T, dt)[1] == 0        # To-Do: change to ValueError, is T 'ganzzahlig' divisible
        super().__init__()
        self.mu    = mu                        
        self.sigma = sigma
        self.r     = r                             
        self.T     = T                             
        self.dt    = dt                           
        self.V_0   = V_0
        self.U_2   = U_2
        
        # Set environment to initial state (t=0, V_t = V_0)
        self.reset()
        
        # Action space (denotes fraction of wealth invested in risky asset, excluding short sales)
        self.action_space = spaces.Box(low=-1, high=1, 
                                       shape=(1,), dtype=np.float32)
        
        # Observation space np.array([t, V_t])
        self.observation_space = spaces.Box(low = np.array([0, 0]), 
                                            high = np.array([self.T, 1e+5]),   #float('Inf')
                                            shape = (2,),
                                            dtype = np.float32)
        
    
    def step(self, action):
        '''Execute one time step within the environment
        
        :params action (float): investment in risky asset
        '''
        
        #assert self.action_space.contains(action)
        
        # Update Wealth (see wealth dynamicy, Inv. Strategies script (by Prof. Zagst) Theorem 2.18):
        # 1) Sample BM increment for one step
        dW_t = np.random.normal(loc=0, scale=math.sqrt(self.dt))
        # 2) Wealth process update via simulation of the exponent
        self.V_t *= np.exp( (self.r + action*(self.mu - self.r) - 0.5*(action**2)*(self.sigma**2)) * self.dt + action*self.sigma*dW_t )                           

        self.t += self.dt                    # updating time-step
        
        done = self.t == self.T              # Episode is finished if termination time is reached
        
        reward = 0                           # Reward is zero for each time step t<T
        if done:                             # Reward at termination time R_T = U(V_T)
            reward = self.U_2(self.V_t)
            
        # Additional info (not used for now)    
        info = {}
            
        return self._get_obs(), reward, done, info          
    
    
    def _get_obs(self):
        return np.array([self.t, self.V_t], dtype=np.float32)
    
            
    def reset(self):
        '''Reset the state of the environment to an initial state'''
        
        self.t   = 0                                          # setting time to zero
        if isinstance(self.V_0, tuple):
            # draw initial wealth uniform from the specified interval
            self.V_t = np.random.uniform(low=self.V_0[0], high=self.V_0[1])
        else:
            # deterministic initial wealth
            self.V_t = self.V_0  
            
        return self._get_obs()
    
    
    #def render(self, mode='human', close=False):
    # Render the environment to the screen
    #...