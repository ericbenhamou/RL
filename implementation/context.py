# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5  17:15:44 2018

@author: eric.benhamou, david.sabbagh
"""


"""
Mdp context to keept track of data
"""
class Mdp_Context:
	def __init__(self):
		self.initialize()
		self.clear()

	"""
	Initialize variables
	"""
	def initialize(self):
		 self.state = 0
		 self.previous_action = 0
		 self.action = 0
		 self.reward = 0
		 self.next_state = 0
		 self.terminal=False
		 self.next_action = 0

		
	"""
	Clear array
	"""
	def clear(self):
		 self.states = []
		 self.actions = []
		 self.rewards = [] 
		 self.terminals = []
		 self.next_states = []
		 self.next_actions = []

	"""
	Append data
	"""
	def save_data(self):
		 self.states.append(self.state)
		 self.actions.append(self.previous_action)
		 self.rewards.append(self.reward)
		 self.terminals.append(self.terminal)
		 self.next_states.append(self.next_state)
		 self.next_actions.append(self.action)


	