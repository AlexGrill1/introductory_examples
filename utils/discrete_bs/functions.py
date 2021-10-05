import numpy as np
import scipy
import scipy.optimize
from collections import defaultdict




def structure_preserving_update(Q, actions, step_size, func, initial_params, T, dt):
    '''Updates the action-value function Q by fitting parameters of function func to Q-table and updating all Q-values
       by a step towards the fitted values.
       
    Args:
    - Q:[dict] Action-value function
    - actions:[np.array] possible risky asset allocations
    - step_size:[float] step_size parameter for an update step towards the fitted values
    - func:[callable] parametrised family of functions used for fitting
    - initial_params:[list[float]] List of initial parameters for the optimisation step (must match the nr. of parameters of func)
    - T:[float] investment horizon
    - dt:[float] times step size
    
    Returns:
    - Q:[dict] Update Action-value function
    '''
    
    # Change dict representation to list representation (needed for scipy.optimize)
    Q_list = _dict_to_list(Q, actions)
    
    # Remove highest / lowest wealth levels from action-value function (wealth levels are set to lower / upper limit, not midpoints)
    tData, aData, vData, qData = _rm_high_low_wealths_from_list(Q_list)
    
    # Fit function
    fittedParameters, pcov = scipy.optimize.curve_fit(func, [tData, aData, vData, T*np.ones(len(tData)), dt*np.ones(len(tData))], qData, p0 = initial_params)
    
    # Fit action-values at grid points
    fittedQValues = func([tData, aData, vData, T*np.ones(len(tData)), dt*np.ones(len(tData))], *fittedParameters)
    
    # difference between predicted values and actual values
    Q_diff = fittedQValues - qData
    
    # Update Q-values
    qData += step_size*Q_diff
    
    # Transform lists to dictionary to dictionary
    Q_new_dict = _lists_to_dict(tData, vData, qData)
    
    # Add states with highest lowest wealths (no change)
    old_keys = set(Q.keys())
    new_keys = set(Q_new_dict.keys())
    difference = list(old_keys - new_keys)
    
    for key in difference:
        Q_new_dict[key] = Q[key]
    
    return Q_new_dict, fittedParameters


def _dict_to_list(Q, actions):
    '''Converts the action-value table Q from a dictionary to a list of four numpy arrays.
    
    Args:
    - Q:[dict] Q-Table
    - actions:[np.array] possible risky asset allocations
    
    Returns a list of four np.arrays:
        - time
        - action
        - wealth
        - Q-value
    '''
    
    tData = np.array([t for t,_ in Q.keys()]).repeat(len(actions))
    aData = np.array(list(actions)*(len(tData)//len(actions)))
    vData = np.array([V for _,V in Q.keys()]).repeat(len(actions))
    qData = np.array([values for values in Q.values()]).reshape(len(tData))
    
    return [tData, aData, vData, qData]


def _lists_to_dict(tData, vData, qData):
    '''Creates the action-value as dictionary from three lists.
    
    Args:
    - tData[list[float]]: time values
    - vData[list[float]]: wealth data
    - qData[list[float]]: action values
    
    Returns:
    - Q[dict]: keys are tuples (t,v) and values are lists of length nr. of actions with respective action-values.
    '''
    
    # Create dict keys as tuples (t,v) -> list of tuples
    keys = list(dict.fromkeys(zip(tData, vData)))
    
    # Create dict values as action-value per action -> list of np.arrays of length nr. actions
    values = np.split(np.array(qData), indices_or_sections = len(keys))
    
    # Create dict
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for key, value in list(zip(keys, values)):
        Q[key] = value
    
    return Q


def _rm_high_low_wealths_from_list(Q_list):
    '''Reduces the Q-value list by their highest and lowest values for function fitting. 
       (Hightest and lowest buckets are not accurate.)
       
    Args:
    - Q_list:[list] list of four lists t, a, v, q from dict_to_list
    
    Returns a list of four np.arrays:
        - time
        - action
        - wealth
        - Q-value
    '''
    
    # Unpack values
    tData, aData, vData, qData = Q_list
    
    # Remove lowest/highest wealth values
    tData = tData[(vData!=min(vData)) & (vData!=max(vData))]
    aData = aData[(vData!=min(vData)) & (vData!=max(vData))]
    qData = qData[(vData!=min(vData)) & (vData!=max(vData))]
    vData = vData[(vData!=min(vData)) & (vData!=max(vData))]
    
    return [tData, aData, vData, qData]