from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time
import json
from past.utils import old_div
import MalmoPython


#############################################################################################################
# Q-Learning for modified tutorial_6.py in the project malmo python examples. Some code is taken from
# there implementation since it is useful in setting up the episode each time, as well as, ensuring that
# the episode has started so that a real observation was found.
#
#
# Written by: 
#             Marc Brittain
#                08/2018
#
#
#############################################################################################################


# Wanted a function to convert to x,z coordinate system instead of carrying this function around
# the majority of this function was build into the project malmo file. I wrapped it up into a file and made
# sure the output was integer values

def convert2XZ(self,world_state):
    """Attempts to extract the current x, z position in minecraft given the current world state"""
    
    # Try statement is to see if there is a valid observation. If not then it returns 0 and we know to try again
    try:
        obs_text = world_state.observations[-1].text
    except:
        return 0

    # Otherwise we can load the text and see if there is x, z in the observation
    obs = json.loads(obs_text) # most recent observation
    if not u'XPos' in obs or not u'ZPos' in obs:

        return 0
    
    # return the integer values corresponding to x and z

    return (int(obs[u'XPos']), int(obs[u'ZPos']))




class Agent:
    def __init__(self,action_size):
        
        # Q-table will be a dictionary because I do not assume that I know the entire dimension of the minecraft
        # arena. I will just add new states as we go.
        self.table = {}
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        
        # setting the minecraft actions here. I have hard coded 1 into each of these action which is moving 1 block
        self.minecraft_actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.learning_rate = 0.1
        
        self.prev_s = None
        self.prev_a = None
        
        
    # Here is the update Q table function. The formula below can be found in many different textbooks on q-learning.
    def update_Q(self,t):
        """given a transition (s,a,r,sp), update the Q-table"""
        
        s,a,r,sp = t
        
        # this is to see if the next state is in the table yet
        try:
            greatestQ = max(self.table[sp])

        except:
            greatestQ = 0
        
        # if the current state is not in the list of keys then let's add it to the q-table
        if not s in self.table.keys():
            self.table[s] = np.zeros(self.action_size)
        
        # finally, the main update 
        self.table[s][a] += self.learning_rate*(r + self.gamma*greatestQ-self.table[s][a])



    # action implementation for the agent
    def act(self, state,agent_host,current_r):
        
        # simple epsilon strategy for the agent. It is set to a low epsilon for standard q-learning
        if random.random() <= self.epsilon:
            # pro tip: random module is faster than numpy random generator so use this is practice
            a = random.randrange(self.action_size)
            # this is to send the command to minecraft
            agent_host.sendCommand(self.minecraft_actions[a])


        else:
            if not state in self.table.keys():
                self.table[state] = np.zeros(self.action_size)
            a = np.argmax(self.table[state])  # returns action
            agent_host.sendCommand(self.minecraft_actions[a])

        self.update_Q([self.prev_s,self.prev_a,current_r,state])

        self.prev_s = state
        self.prev_a = a

        return current_r



    def run(self, agent_host):
        """run the agent on the world"""
        
        # initialize total reward for one trial
        total_reward = 0
        
        # initialize our current reward
        current_r = 0

        # this is for waiting for the first observation
        starter = False

        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:

            count = 0
            while not starter:
                    world_state = agent_host.getWorldState()
                    s = convert2XZ(self,world_state)
                    if s != 0:
                        starter = True
    

            starter = False
            
            # recieve the reward of this state
            current_r = sum(r.getValue() for r in world_state.rewards)
            
            # update our agent with the action
            total_reward += self.act(s,agent_host,current_r)
            
            
            # I really do not like this sleep here...When setting lower than this, MineCraft would get stuck and I 
            # would have to restart. I believe it has to do with waiting for another quality observation...
            # I will look into this more in future episodes where time is more demanding
            
            time.sleep(0.25)


            if not world_state.is_mission_running:
                break


        return total_reward


    
### Defining constants here

EPISODES = 200

# -- set up the mission -- #
mission_file = './episode_1.xml'

# defining the action space dimension of the agent
agent = Agent(4)

cumulative_rewards = []

count = 0

# I want 50% of episodes at 0.1, 25% at 0.05 and 25% at 0.001
epsilon_vals = [0.1,0.05,0.001]
change_epsilon = EPISODES//4

epsilon_count = 0
agent.epsilon = epsilon_vals[epsilon_count]





### Next Section is from tutorial6.py in the project Malmo Python examples. I have made minor changes to ### improve the readability and the performance of the code. Too many print statements for me :)

### The corresponding .xml file has been renamed, but is the same as well

# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# ------------------------------------------------------------------------------------------------

# Tutorial sample #6: Discrete movement, rewards, and learning

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998

##### <------ Nothing edited here through next break

if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk



if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

agent_host = MalmoPython.AgentHost()


try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)
    
##### <------- end break



with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
# add 20% holes for interest
for x in range(1,4):
    for z in range(1,13):
        if random.random()<0.1:
            my_mission.drawBlock( x,45,z,"lava")
            
            

max_retries = 3

if agent_host.receivedArgument("test"):
    num_repeats = 1
else:
    num_repeats = EPISODES



for i in range(num_repeats):

    print()
    print('Episode {} of {}'.format( i+1, num_repeats ))

    my_mission_record = MalmoPython.MissionRecordSpec()

    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2.5)


    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    # -- run the agent in the world -- #

    # This is for the first change
    if i % change_epsilon*2 and epsilon_count == 0:
        epsilon_count += 1
        agent.epsilon = epsilon_vals[epsilon_count]

    # this if for changing epsilon after the first change
    if i % change_epsilon and epsilon_count > 0:
        epsilon_count += 1

        # Maybe our integer division made the change an episode early. This try statement
        # will prevent any error from being potentially thrown due to this

        try:
            agent.epsilon = epsilon_vals[epsilon_count]
        except:
            agent.epsilon = epsilon_vals[-1]


    cumulative_reward = agent.run(agent_host)


    print('Cumulative reward: {}'.format(cumulative_reward))
    cumulative_rewards += [ cumulative_reward ]

    # -- clean up -- #
    time.sleep(0.5) # (let the Mod reset)
