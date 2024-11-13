"""
 test.py

 Python script implementing test cases for debugging.

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

import pickle

import numpy as np
from ComputeExpectedStageCosts import compute_expected_stage_cost
from ComputeTransitionProbabilities import compute_transition_probabilities
from Constants import Constants
from Solver import solution
import matplotlib.pyplot as plt
from utils import *
import time


if __name__ == "__main__":
    n_tests = 4
    for i in range(n_tests):
        print("-----------")
        print("Test " + str(i))
        with open("tests/test" + str(i) + ".pkl", "rb") as f:
            loaded_constants = pickle.load(f)
            for attr_name, attr_value in loaded_constants.items():
                if hasattr(Constants, attr_name):
                    setattr(Constants, attr_name, attr_value)

        file = np.load("tests/test" + str(i) + ".npz")

        # Begin tests
        t_start1 = time.time()
        P = compute_transition_probabilities(Constants)
        t_end1 = time.time() - t_start1
        if not np.all(
            np.logical_or(np.isclose(P.sum(axis=1), 1), np.isclose(P.sum(axis=1), 0))
        ):
            print(
                "[ERROR] Transition probabilities do not sum up to 1 or 0 along axis 1!"
            )

        t_start2 = time.time()
        Q = compute_expected_stage_cost(Constants)
        t_end2 = time.time() - t_start2
        passed = True
        if not np.allclose(P, file["P"], rtol=1e-4, atol=1e-7):
            print("Wrong transition probabilities")
            passed = False
        else:
            print("Correct transition probabilities")

        if not np.allclose(Q, file["Q"], rtol=1e-4, atol=1e-7):
            print("Wrong expected stage costs")
            passed = False
        else:
            print("Correct expected stage costs")

        # normal solution
        t_start3 = time.time()
        [J_opt, u_opt] = solution(P, Q, Constants)
        t_end3 = time.time()-t_start3
        if not np.allclose(J_opt, file["J"], rtol=1e-4, atol=1e-7):
            print("Wrong optimal cost")
            diff = J_opt - file["J"]
            passed = False
        else:
            print("Correct optimal cost")
            print("Total Elapsed Time:",t_end1 + t_end2 + t_end3)
            print("Probabilities Elapsed Time:", t_end1)
            print("Expected Reward Elapsed Time",t_end2)
            print("Solving Elapsed Time:", t_end3)

    print("-----------")
