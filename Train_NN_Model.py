
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

import random
import numpy as np
import math
############### Defining Helper functions ########################



#Return winner -1 for X or 1 for O and 0 for draw
def get_winer(state):
	total_reward = 0
	if (state[0] == state[1] == state[2]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[3] == state[4] == state[5]) and not state[3] == 0:
		total_reward = state[3]	
	elif (state[6] == state[7] == state[8]) and not state[6] == 0:
		total_reward = state[6]	
	elif (state[0] == state[3] == state[6]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[1] == state[4] == state[7]) and not state[1] == 0:
		total_reward = state[1]	
	elif (state[2] == state[5] == state[8]) and not state[2] == 0:
		total_reward = state[2]	
	elif (state[0] == state[4] == state[8]) and not state[0] == 0:
		total_reward = state[0]	
	elif (state[2] == state[4] == state[6]) and not state[2] == 0:
		total_reward = state[2]
	return total_reward

try:
	model_X = load_model('tic_tac_toe_X.h5')
	model_O = load_model('tic_tac_toe_O.h5')
	print('Pre-existing model found... loading data.')
except:
	pass

#Hot Encode Tic Tac Toe Board 
def one_hot(state):
	current_state = []
	for square in state:
		if square == 0:
			current_state.append(1)
			current_state.append(0)
			current_state.append(0)
		elif square == 1:
			current_state.append(0)
			current_state.append(1)
			current_state.append(0)
		elif square == -1:
			current_state.append(0)
			current_state.append(0)
			current_state.append(1)
	return current_state

#Calculate Q-Values for each state at 1 game after that we create a list of games that contain Q-Values for each game at each state 
#Train Our NN with this Data
def get_Q_values(games, model_X, model_O):
	global x_train
	xt = 0
	ot = 0
	dt = 0
	states = []
	q_values = []
	states_2 = []
	q_values_2 = []

	for game in games:
		total_reward = get_winer(game[len(game) - 1])
		if total_reward == -1:
			ot += 1
		elif total_reward == 1:
			xt += 1
		else:
			dt += 1
		

		for i in range(0, len(game) - 1):
			if i % 2 == 0:
				for j in range(0, 9):
					if not game[i][j] == game[i + 1][j]:
						reward = np.zeros(9)
						reward[j] = total_reward*(alpha**(math.floor((len(game) - i) / 2) - 1))
						# print(reward)
						states.append(game[i].copy())
						q_values.append(reward.copy())
			else:
				for j in range(0, 9):
					if not game[i][j] == game[i + 1][j]:
						reward = np.zeros(9)
						reward[j] = -1*total_reward*(alpha**(math.floor((len(game) - i) / 2) - 1))
						# print(reward)
						states_2.append(game[i].copy())
						q_values_2.append(reward.copy())

	if x_train:
		zipped = list(zip(states, q_values))
		random.shuffle(zipped)
		states, q_values = zip(*zipped)
		new_states = []
		for state in states:
			new_states.append(one_hot(state))
		model_X.fit(np.asarray(new_states), np.asarray(q_values), epochs=4, batch_size=len(q_values), verbose=1)
		model_X.save('tic_tac_toe_X.h5')
		del model_X	
	else:
		zipped = list(zip(states_2, q_values_2))
		random.shuffle(zipped)
		states_2, q_values_2 = zip(*zipped)
		new_states = []
		for state in states_2:
			new_states.append(one_hot(state))
		model_O.fit(np.asarray(new_states), np.asarray(q_values_2), epochs=4, batch_size=len(q_values_2), verbose=1)
		model_O.save('tic_tac_toe_O.h5')
		del model_O
		model_O = load_model('tic_tac_toe_O.h5')
		print(xt/20, ot/20, dt/20)
	x_train = not x_train




############### Main Script ########################
alpha = .7
x_train = True



model_X = Sequential()
model_X.add(Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_X.add(Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_X.add(Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_X.add(Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_X.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_X.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model_O = Sequential()
model_O.add(Dense(units=130, activation='relu', input_dim=27, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_O.add(Dense(units=250, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_O.add(Dense(units=140, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_O.add(Dense(units=60, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
model_O.add(Dense(9, kernel_initializer='random_uniform', bias_initializer='zeros'))
model_O.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

mode = 'training'

while True:
	board = [0, 0, 0, 0,  0, 0, 0, 0, 0]
	games = []
	current_game = []
	if mode == 'training':
		total_games = 3000
		exploration_rate = .7
		for i in range(0, total_games):
			playing = True
			nn_turn = True
			c = 0
			board = [0, 0, 0, 0,  0, 0, 0, 0, 0]
			current_game = []
			current_game.append(board.copy())
			nn_board = board
			while playing:
				if nn_turn:
					if random.uniform(0, 1) <= exploration_rate:
						choosing = True
						while choosing:
							c = random.randint(0, 8)
							if board[c] == 0:
								choosing = False
								board[c] = 1
								current_game.append(board.copy())
					else:
						pre = model_X.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
						highest = -1000
						num = -1
						for j in range(0, 9):
							if board[j] == 0:
								if pre[j] > highest:
									highest = pre[j].copy()
									num = j
						choosing = False
						board[num] = 1
						current_game.append(board.copy())
				else:
					if random.uniform(0, 1) <= exploration_rate:
						choosing = True
						while choosing:
							c = random.randint(0, 8)
							if board[c] == 0:
								choosing = False
								board[c] = -1
								current_game.append(board.copy())
					else:
						pre = model_O.predict(np.asarray([one_hot(board)]), batch_size=1)[0]
						highest = -1000
						num = -1
						for j in range(0, 9):
							if board[j] == 0:
								if pre[j] > highest:
									highest = pre[j].copy()
									num = j
						choosing = False
						board[num] = -1
						current_game.append(board.copy())
				playable = False
				for square in board:
					if square == 0:
						playable = True
				if not get_winer(board) == 0:
					playable = False
				if not playable:
					playing = False
				nn_turn = not nn_turn
			games.append(current_game)
		get_Q_values(games, model_X, model_O)
	