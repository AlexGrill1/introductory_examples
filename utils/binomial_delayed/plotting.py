import numpy as np
import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt

from envs.binomial_tree import encode_wealth, encode_action, decode_action



def plot_q_values(Q, actions):
    '''Creates a 3d Wireframe plot of the Q-value function for each state-action pair and adds the predicted action (i.e. argmax_a Q(s,a)'''
    
    
    def fun(x, y, Q, actions):
        '''Help function used in plot_q_values'''
        # Returns the Q-values for each state-action pair at time step 1.
        return np.array([Q[(1,wealth)][encode_action(action, actions)] for action, wealth in zip(x,y)])
    
    
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
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

    ax.set_xlabel('investment in risky asset', fontsize=15)
    ax.set_ylabel('wealth', fontsize=15)
    ax.set_zlabel('Q-values', fontsize=15)
    ax.scatter(predicted_actions, wealths, predicted_Q_values, zdir="z", s=40, c="red", alpha=1, label="Predicted Actions")
    plt.title("Learned action-value surface at time t={} (delayed rewards)".format(1), fontsize=20)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.show()
    
    

def plot_learned_vs_optimal_policy(Q, actions):
    '''Plots the learned policy derived from the action-value function Q vs. the optimal policy for sqrt utility.
    
    Args:
    :params Q[dict]: The Q-Table.
    :params actions[np.array]: np.array containing the possible investment choices.
    '''
    
    # get the observed wealth levels for each time
    wealth_levels = sorted([wealth for t, wealth in Q.keys() if t == 1])
    # Derives the investment choice from the action-value function Q for the given state
    predicted_actions = [decode_action(np.argmax(Q[(1, wealth)]), actions) for wealth in wealth_levels]

        
    # Plots the learned policy
    plt.figure(figsize=(20,10))
    plt.plot(wealth_levels, predicted_actions, label="learned")
    # Plots the optimal policy for sqrt utility
    plt.plot(wealth_levels, 0.6923*np.ones(len(wealth_levels)), "-.", label="optimal")
    plt.ylim(-.01, 1.01)
    plt.title("Learned policy vs. optimal policy (at time t={})".format(1), fontsize=25)
    plt.xlabel("wealth", fontsize=20)
    plt.ylabel("risky asset allocation", fontsize=20)
    plt.legend(fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
        
    plt.show()