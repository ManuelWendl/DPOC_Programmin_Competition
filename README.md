# Programming Competition 

This repo contains the submitted code for the Dynamic Programming and Optimal Control Class at ETH Zürich in Autumn Semester 2024/25 taught by Prof. Dr. Raffaello D'Andrea. 
This submission reached 5th rank in the course wide competition. 
It uses different approaches to solve a grid world MDP the shortest path problem, including value and policy iteration and its different modifications. The fastest implementation is based on policy iteration and solves a reduced linear system of equations in each iteration. It only consists of valid reachable states, which reduces the computational demand. 
The main.py, Constants.py, test.py, utils.py and visualization.py files were provided. 
![Alt text](./example.png?raw=true)
