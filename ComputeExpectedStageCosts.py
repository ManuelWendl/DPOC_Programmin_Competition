"""
 ComputeExpectedStageCosts.py

 Python function template to compute the expected stage cost.

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


def compute_expected_stage_cost(Constants):
    """Computes the expected stage cost for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L)
    """
     # TODO fill the transition probability matrix P here
    try:
        Q = Constants.TIME_COST + Constants.THRUSTER_COST * np.sum(np.abs(Constants.INPUT_SPACE),1)[None,:] + Constants.DRONE_COST * Constants.Pstart
        Q[Constants.extra_states,:] = 0
    except: 
        Q = np.ones((Constants.K, Constants.L)) * np.inf

        # Set where swan is in the same location as in the robot: [x_r,y_r,x_s,y_s] -> xr + yr*M + xs*M*N + ys*M*N*M
        total_states = set(range(Constants.K))
        illegal_swan_states = set([i+j*Constants.M+i*Constants.M*Constants.N+j*Constants.M*Constants.N*Constants.M for i in range(Constants.M) for j in range(Constants.N)])
        illegal_drone_states = set([coords[0]+coords[1]*Constants.M+i*Constants.M*Constants.N+j*Constants.M*Constants.N*Constants.M for coords in Constants.DRONE_POS for i in range(Constants.M) for j in range(Constants.N)])
        goal_states = set([Constants.GOAL_POS[0]+Constants.GOAL_POS[1]*Constants.M+i*Constants.M*Constants.N+j*Constants.M*Constants.N*Constants.M for i in range(Constants.M) for j in range(Constants.N)])
        extra_states = set.union(goal_states,illegal_drone_states,illegal_swan_states)

        def calculate_swan_move(robot_pos, swan_pos):
            theta = np.arctan2(robot_pos[1]-swan_pos[1], robot_pos[0]-swan_pos[0])
            if theta >= -np.pi/8 and theta < np.pi/8:
                return np.array([1,0])
            elif theta >= np.pi/8 and theta < 3*np.pi/8:
                return np.array([1,1])
            elif theta >= 3*np.pi/8 and theta < 5*np.pi/8:
                return np.array([0,1])
            elif theta >= 5*np.pi/8 and theta < 7*np.pi/8:
                return np.array([-1,1])
            elif theta >= 7*np.pi/8 or theta < -7*np.pi/8:
                return np.array([-1,0])
            elif theta >= -7*np.pi/8 and theta < -5*np.pi/8:
                return np.array([-1,-1])
            elif theta >= -5*np.pi/8 and theta < -3*np.pi/8:
                return np.array([0,-1])
            elif theta >= -3*np.pi/8 and theta < -np.pi/8:
                return np.array([1,-1])
            else:
                return np.array([0,0])

        def verify_state(state):
            if state[0] < 0 or state[0] >= Constants.M or state[1] < 0 or state[1] >= Constants.N:
                return True
            return False

        def check_drone_collision(start_state,end_state):
            for point in bresenham(start_state,end_state):
                if (point[0] + point[1]*Constants.M) in illegal_drone_states:
                    return True
            return False

        def check_drone_collision_simple(end_state):
            if (end_state[0] + end_state[1]*Constants.M) in illegal_drone_states:
                    return True
            return False


        # Set the transition probabilities for the goal states
        for i in list(extra_states):
            Q[i,:] = 0.

        # Itterate over all legal states
        for i in list(total_states-extra_states):
            coords = Constants.STATE_SPACE[i]

            # Compute the next state
            w_curr = Constants.FLOW_FIELD[coords[0],coords[1],:]
            p_curr = Constants.CURRENT_PROB[coords[0],coords[1]]

            swan_move = calculate_swan_move(coords[:2], coords[2:])

            for l in range(Constants.L):

                next_state = coords[:2] + Constants.INPUT_SPACE[l]
                next_state_p = next_state + w_curr
                next_state_pm1 = next_state

                p_next_state_p_p, p_next_state_p_pm1 = p_curr * Constants.SWAN_PROB, p_curr * (1-Constants.SWAN_PROB)
                p_next_state_pm1_p, p_next_state_pm1_pm1 = (1-p_curr) * Constants.SWAN_PROB, (1-p_curr) * (1-Constants.SWAN_PROB)
                p_start = 0

                # Compute next state of swan
                next_state_swan_p = coords[2:] + swan_move
                next_state_swan_pm1 = coords[2:]

                # Check if the next state is valid
                if verify_state(next_state_pm1) or check_drone_collision_simple(next_state_pm1):
                    p_start += p_next_state_pm1_p + p_next_state_pm1_pm1
                else:
                    if all(next_state_pm1==next_state_swan_p):
                        p_start += p_next_state_pm1_p 
                    elif all(next_state_pm1==next_state_swan_pm1):
                        p_start += p_next_state_pm1_pm1

                if verify_state(next_state_p) or check_drone_collision(coords[:2], next_state_p):
                    p_start += p_next_state_p_p + p_next_state_p_pm1
                else:
                    if all(next_state_p==next_state_swan_p):
                        p_start += p_next_state_p_p

                    elif all(next_state_p==next_state_swan_pm1):
                        p_start += p_next_state_p_pm1

                Q[i,l] = Constants.TIME_COST + Constants.THRUSTER_COST * np.sum(np.abs(Constants.INPUT_SPACE[l])) + Constants.DRONE_COST * p_start

    return Q
