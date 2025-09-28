import os

os.system("echo 'Epsilon-greedy Monte Carlo:'")
os.system("python ./eps-greedy_MC_learning.py")

os.system("echo 'Off-policy Monte Carlo:'")
os.system("python ./off-policy_MC_learning.py")

os.system("echo 'SARSA TD learning:'")
os.system("python ./SARSA_TD_learning.py")

os.system("echo 'Q learning:'")
os.system("python ./QL_TD_learning.py")

os.system("echo 'Deep Q learning:'")
os.system("python ./DQL.py")