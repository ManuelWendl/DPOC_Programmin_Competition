"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

 Dynamic Programming and Optimal Control
 Fall 2024
 Programming Exercise

 Contact: Antonio Terpin aterpin@ethz.ch

 Authors: Maximilian Stralz, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np
from utils import *


def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """

    # TODO fill the transition probability matrix P here

    # Set where swan is in the same location as in the robot: [x_r,y_r,x_s,y_s] -> xr + yr*M + xs*M*N + ys*M*N*M
    M = Constants.M
    N = Constants.N
    MN = M*N
    MNM = MN*M

    illegal_start_index = {Constants.START_POS[0]+Constants.START_POS[1]*M+Constants.START_POS[0]*MN+Constants.START_POS[1]*MNM}
    start_idx = list({Constants.START_POS[0]+Constants.START_POS[1]*M+j*MN+k*MNM for j in range(M) for k in range(N)} - illegal_start_index)

    illegal_drone_states, extra_states, goal_state = {coords[0]+coords[1]*M for coords in Constants.DRONE_POS}, [], Constants.GOAL_POS[0]+Constants.GOAL_POS[1]*M

    P, Pstart = np.zeros((Constants.K, Constants.K, Constants.L)), np.zeros((Constants.K, Constants.L))

    def calculate_swan_move(dx, dy):
        angle = np.arctan2(dy, dx)
        if -np.pi/8 <= angle < np.pi/8:
            return np.array([1, 0])
        elif np.pi/8 <= angle < 3*np.pi/8:
            return np.array([1, 1])
        elif 3*np.pi/8 <= angle < 5*np.pi/8:
            return np.array([0, 1])
        elif 5*np.pi/8 <= angle < 7*np.pi/8:
            return np.array([-1, 1])
        elif 7*np.pi/8 <= angle or angle < -7*np.pi/8:
            return np.array([-1, 0])
        elif -7*np.pi/8 <= angle < -5*np.pi/8:
            return np.array([-1, -1])
        elif -5*np.pi/8 <= angle < -3*np.pi/8:
            return np.array([0, -1])
        elif -3*np.pi/8 <= angle < -np.pi/8:
            return np.array([1, -1])
        else:
            return np.array([0, 0])        
        
    swan_move_cache = {}
    def get_swan_move(robot, swan):
        dx = robot[0] - swan[0]
        dy = robot[1] - swan[1]
        key = (dx+M, dy+N)
        if key not in swan_move_cache:
            swan_move_cache[key] = calculate_swan_move(dx, dy)
        return swan_move_cache[key]
        
    def verify_state(state):
        if not (0 <= state[0] < M and 0 <= state[1] < N):
            return True
        return False
    
    def check_drone_collision(start,end):
        x0 = start[0]
        y0 = start[1]
        x1 = end[0]
        y1 = end[1]

        dx = x1 - x0
        dy = y1 - y0

        x_sign = 1 if dx > 0 else -1 if dx < 0 else 0
        y_sign = 1 if dy > 0 else -1 if dy < 0 else 0
        
        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = x_sign, 0, 0, y_sign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, y_sign, x_sign, 0

        D = 2 * dy - dx
        y = 0

        for x in range(dx + 1):
            px = x0 + x * xx + y * yx
            py = y0 + x * xy + y * yy
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy

            if (px + py*M) in illegal_drone_states:
                return True

        return False
    
    def check_drone_collision_simple(end_state):
        if (end_state[0] + end_state[1]*M) in illegal_drone_states:
                return True
        return False

    # Itterate over all legal states
    for i in range(Constants.K):
        indx_robot = i % MN
        if indx_robot in illegal_drone_states:
            extra_states.append(i)
            continue
        if indx_robot == goal_state:
            extra_states.append(i)
            continue
        indx_swan = i // MN
        if indx_robot == indx_swan:
            extra_states.append(i)
            continue

        coords = Constants.STATE_SPACE[i]
        swan_move, w_curr, p_curr = get_swan_move(coords[:2], coords[2:]), Constants.FLOW_FIELD[coords[0],coords[1],:],Constants.CURRENT_PROB[coords[0],coords[1]] 
        p_next_state_p_p, p_next_state_p_pm1, p_next_state_pm1_p, p_next_state_pm1_pm1 = p_curr * Constants.SWAN_PROB, p_curr * (1-Constants.SWAN_PROB), (1-p_curr) * Constants.SWAN_PROB, (1-p_curr) * (1-Constants.SWAN_PROB)

        for l in range(Constants.L):
            # Compute the next state
            next_state = coords[:2] + Constants.INPUT_SPACE[l]
            next_state_p, next_state_pm1, p_start, next_state_swan_p, next_state_swan_pm1 = next_state + w_curr, next_state, 0, coords[2:] + swan_move, coords[2:]

            # Compute next state of swan
            indx_pm1, indx_p, indx_swan_p, indx_swan_pm1 = next_state_pm1[0] + next_state_pm1[1]*M, next_state_p[0] + next_state_p[1]*M, next_state_swan_p[0] + next_state_swan_p[1]*M, next_state_swan_pm1[0] + next_state_swan_pm1[1]*M

            # Check if the next state is valid
            if verify_state(next_state_pm1) or check_drone_collision_simple(next_state_pm1):
                p_start += p_next_state_pm1_p + p_next_state_pm1_pm1
            else:
                if indx_pm1 == indx_swan_p:
                    p_start += p_next_state_pm1_p 
                else: 
                    idx_pm1_p = indx_pm1 + indx_swan_p*MN
                    P[i,idx_pm1_p,l] += p_next_state_pm1_p

                if indx_pm1 == indx_swan_pm1:
                    p_start += p_next_state_pm1_pm1
                else:
                    idx_pm1_pm1 = indx_pm1 + indx_swan_pm1*MN
                    P[i,idx_pm1_pm1,l] += p_next_state_pm1_pm1
                
            if verify_state(next_state_p) or check_drone_collision(coords[:2], next_state_p):
                p_start += p_next_state_p_p + p_next_state_p_pm1
            else:
                if indx_p == indx_swan_p:
                    p_start += p_next_state_p_p
                else:
                    idx_p_p = indx_p + indx_swan_p*MN
                    P[i,idx_p_p,l] += p_next_state_p_p

                if indx_p == indx_swan_pm1:
                    p_start += p_next_state_p_pm1
                else:
                    idx_p_pm1 = indx_p + indx_swan_pm1*MN
                    P[i,idx_p_pm1,l] += p_next_state_p_pm1
               
            Pstart[i,l] = p_start
            P[i,start_idx,l] += p_start/(MN-1)

    Constants.extra_states, Constants.Pstart = extra_states, Pstart
    return P
