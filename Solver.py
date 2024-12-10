"""
 Solver.py

 Python function template to solve the stochastic
 shortest path problem.

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
from scipy.linalg import solve
from utils import *


def solution(P, Q, Constants):
    """Computes the optimal cost and the optimal control input for each
    state of the state space solving the stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming;
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        Q  (np.array): A (K x L)-matrix containing the expected stage costs of all states
                       in the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP
        np.array: The optimal control policy for the stochastic SPP

    """

    J_opt = np.zeros(Constants.K)
    u_opt = np.zeros(Constants.K)

    # TODO implement Value Iteration, Policy Iteration,
    #      Linear Programming or a combination of these

    solvers = ['VI','PI_VI','PI_SLE','PI_SLE_Reduced','LP']
    solver = solvers[3]

    if solver == 'VI':
        epsilon = 1e-4
        rel_diff = np.inf
        while rel_diff >= epsilon:
            J_opt_new = np.min(Q+np.einsum('ijk,j->ik',P,J_opt),axis=1)
            rel_diff = np.max(np.abs(J_opt_new - J_opt))
            J_opt = J_opt_new
        u_opt = np.argmin(Q+np.einsum('ijk,j->ik',P,J_opt),axis=1)

    if solver == 'PI_VI':
        epsilon = 1e-6
        old_policy = np.ones_like(u_opt)*4
        while any(old_policy != u_opt):
            old_policy = u_opt
            u_opt = np.argmin(Q+np.einsum('ijk,j->ik',P,J_opt),axis=1)
            rel_diff = np.inf
            for i in range(10):
                J_opt_new = Q[np.arange(Constants.K),u_opt.astype(int)]+np.einsum('ij,j->i',P[np.arange(Constants.K),:,u_opt.astype(int)],J_opt)
                rel_diff = np.max(np.abs(J_opt_new - J_opt))
                J_opt = J_opt_new
        while rel_diff >= epsilon:
                J_opt_new = Q[np.arange(Constants.K),u_opt.astype(int)]+np.einsum('ij,j->i',P[np.arange(Constants.K),:,u_opt.astype(int)],J_opt)
                rel_diff = np.max(np.abs(J_opt_new - J_opt))
                J_opt = J_opt_new

    if solver == 'PI_SLE':
        old_policy = np.ones_like(u_opt)
        I = np.eye(Constants.K)
        while any(old_policy != u_opt):
            old_policy = u_opt
            u_opt = np.argmin(Q+np.einsum('ijk,j->ik',P,J_opt),axis=1)
            J_opt = np.linalg.solve(I-P[np.arange(Constants.K),:,u_opt.astype(int)],Q[np.arange(Constants.K),u_opt.astype(int)]) 

    if solver == 'PI_SLE_Reduced':
        considered_states = list(set(range(Constants.K)) - set(Constants.extra_states))
        J_red = np.zeros(len(considered_states))
        u_red = np.zeros(len(considered_states))
        old_policy = np.ones_like(u_red)
        I = np.eye(len(considered_states))
        P_red = P[np.ix_(considered_states, considered_states, range(P.shape[2]))]
        Q_red = Q[considered_states]
        while any(old_policy != u_red):
            old_policy = u_red
            u_red = np.argmin(Q_red+np.einsum('ijk,j->ik',P_red,J_red),axis=1)
            J_red = np.linalg.solve(I-P_red[np.arange(len(considered_states)),:,u_red.astype(int)],Q_red[np.arange(len(considered_states)),u_red.astype(int)])
        
        J_opt[considered_states] = J_red
        u_opt[considered_states] = u_red

    
    #if solver == 'LP':
    #    # Not functional yet
    #    c = np.ones_like(J_opt)
    #    P_reshaped = P.reshape(Constants.K * Constants.L, Constants.K)
    #    Q_reshaped = Q.reshape(Constants.K * Constants.L,)
    #    eyes = np.repeat(np.eye(Constants.K),Constants.L,axis=0)
    #    A = eyes-P_reshaped

    #    J_opt = sc.optimize.linprog(c,A_ub=A,b_ub=Q_reshaped).x
    #    u_opt = np.argmin(Q+np.einsum('ijk,j->ik',P,J_opt),axis=1)

    return J_opt, u_opt
