
import os
import json
import random

EMPTY_POS = ' '
X_POS = 'X'
O_POS = 'O'

def is_empty_position(i: int, j: int, board: list):
    """
        Checks whether the given position is empty or not in the board.
    """
    return board[i][j] == EMPTY_POS

def get_empty_board(length: int=3):
    """
        Returns an empty square board in the given length.
        The length is supposed to be 3 for the usual tic tac toe game.
    """
    return [[EMPTY_POS for _ in range(length)] for __ in range(length)]

def get_winner(board: list):
    """
        Checks if there's a winner in the board.
        Returns two values, the winner symbol, and a boolean variable.

        The winner symbol is an integer from the POSITIONS class (X or O i.e. 1 or 2).
        The boolean variable is set to True if the board is in a draw state. False otherwise.

        If the winner symbol is equal to EMPTY_POS, 
        it means that there's not winner yet.
    """
    
    is_win_state = lambda lst: len(set(lst))==1 and lst[0]!=EMPTY_POS
    length = len(board)
    # list of possible win states

    # rows
    possible_win_states = [row for row in board]

    # columns
    possible_win_states += [col for col in list(zip(*board))]

    # diagonals
    possible_win_states += [[board[i][i]          for i in range(length)]]
    possible_win_states += [[board[i][length-i-1] for i in range(length)]]

    number_of_empty_positions = len(get_empty_positions(board))

    # go through all possible states and check if there's a winner
    for state in possible_win_states:
        if is_win_state(state):
            winner = state[0]
            is_draw = False
            
            # we found a winner!
            return winner, is_draw

    # tie
    if number_of_empty_positions == 0:
        winner = EMPTY_POS
        is_draw = True
 
        return winner, is_draw

    # there's no winner yet, and the game is not in a draw state yet
    return EMPTY_POS, False

def get_board_state(board: list):
    """
        Return a unique string that represents the state of the current board.
    """
    return ''.join(''.join([cell for cell in row]) for row in board)

def display_board(board):
    """
        Displays the current board on the command line.
    """
    length = len(board)
    bar = "   +---+---+---+"
    print("      0   1   2")
    for i in range(length):
        print(bar)
        print(f" {i} | "+ "".join(board[i][j] + " | " for j in range(length)))
    print(bar)

def get_empty_positions(board: list):
    """
        Returns all empty positions within the board.
    """
    length = len(board)
    empty_positions = []
    for i in range(length):
        for j in range(length):
            if is_empty_position(i, j, board):
                empty_positions.append((i,j))
    
    return empty_positions

def mark_move(i: int, j: int, board: list, symbol: str):
    """
        Marks a move on the board using the given symbol.
    """
    board[i][j] = symbol

def play_against_agent(agent_policy_file):
    board = get_empty_board()
    agent = Agent("Trained Agent", X_POS, exploration_rate=0)
    agent.load_agent_policy(agent_policy_file)

    player = Player("Human", O_POS)

    agent_turn = True
    while True:
        display_board(board)
        current_player = agent if agent_turn else player
        print(f"{current_player.name} turn: ")
        agent_turn = not agent_turn
        action = current_player.get_next_action(board)
        mark_move(action[0], action[1], board, current_player.symbol)
        print()
        # check if there's a winner or if there's a draw
        winner, is_draw = get_winner(board)
        if (is_draw) or (winner != EMPTY_POS):
            break
    
    display_board(board)
    if is_draw:
        print("Draw!")

    else:
        winner_name = agent.name if winner == agent.symbol else player.name
        print(f"{winner_name} wins!")

class Player:
    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol
    
    def get_valid_value(self, board_length, row=True):
        value_name = "row" if row else "col"
        while True:
            try:
                row = int(input(f"Please enter the {value_name} index[0-{board_length-1}]: "))
                assert (row<board_length)
                return row

            except:
                print(f"Invalid row! Please enter a valid {value_name} index!")
    
    def get_next_action(self, board):
        """
            Get an action from the player.
        """
        length = len(board)

        while True:
            i = self.get_valid_value(length)
            j = self.get_valid_value(length, False)
            if not is_empty_position(i, j, board):
                print("The position is not empty! Please pick another position.")
            
            else:
                return i, j

class Agent:
    def __init__(self, name, symbol, exploration_rate=0.45, init_alpha=0.45, init_gamma=0.9):
        # the name of the agent, only used for visualization 
        self.name = name

        # the symbol of the agent (i.e X or Y)
        self.symbol = symbol

        # a sorted list of all the board states that the agent went through during a single game
        # it's used for applying the update rule
        self.board_states = []

        # the policy is basically a dictionary that maps each board state to a probability
        # the 
        self.policy = dict()

        # the probability of exploring other states
        # instead of going with the greedy approach of choosing the best possible state
        self.exploration_rate = exploration_rate

        # set the initial alpha and gamma parameters
        # will be used later to reset the agent
        self.init_alpha = init_alpha
        self.init_gamma = init_gamma
        self.init_exploration_rate = exploration_rate

        # the parameters of the update rule of Q learning
        # Q*(s, a) = Q(s, a) + alpha * (reward + gamma * MAX(Q(s+1, a)) - Q(s, a)) 
        self.alpha = self.init_alpha
        self.gamma = self.init_gamma

    def get_next_action(self, board):
        """
            Gets the next action that the agent shall perform on the given board.
        """
        possible_actions = get_empty_positions(board)

        # check if we should explore rather than exploit
        if random.uniform(0, 1) <= self.exploration_rate:
            # explore, pick a random position as the next action
            return random.choice(possible_actions)

        # exploit, be greedy and look for the optimal move
        value_of_optimal_action = -1e8
        for action in possible_actions:
            # make a copy of the board to test our action on
            board_copy = [[cell for cell in row] for row in board]

            # mark the current action on the board
            mark_move(action[0], action[1], board_copy, self.symbol)

            # get the state of the board copy
            board_state = get_board_state(board_copy)

            # get the value (if exists) of that board copy from the policy
            curr_action_value = self.policy.get(board_state, 0)
            if value_of_optimal_action <= curr_action_value:
                value_of_optimal_action = curr_action_value
                optimal_action = action

        return optimal_action

    def update_policy(self, reward):
        """
            Updates the policy of the agent after a finished game.
        """
        for board_state in self.board_states[::-1]:
            self.policy[board_state] = self.policy.get(board_state, 0)

            self.policy[board_state] += self.alpha * (self.gamma * reward - self.policy[board_state])
            reward = self.policy[board_state]

        # decrease the learning rate and exploration rate after each game so that the agent can converge over time
        self.alpha = max(self.alpha-0.000004, 0.2)
        self.exploration_rate = max(self.exploration_rate-0.000004, 0.1)

    def reset(self):
        """
            Resets the agent values.
        """
        # self.alpha = self.init_alpha
        # self.gamma = self.init_gamma
        # self.exploration_rate = self.init_exploration_rate
        self.board_states = []

    def save_agent_policy(self, filename):
        """
            Save the policy of the agent to a JSON file for later use.
        """
        with open(filename, 'w+') as fh:
            json.dump(self.policy, fh, indent=4)

    def load_agent_policy(self, filename):
        """
            Load the policy of the agent from JSON file.
        """
        with open(filename, 'r') as fh:
            self.policy = json.load(fh)

class AgentTrainer:
    def __init__(self):
        self.board = get_empty_board()

        # create two agents that will compete against each other
        # we will optimize agent1 more on draw games so it can learn better
        # we will only save the policy of agent1 after training
        # agent2 will be a bit smart as well, but not as smart as agent1
        self.agent1 = Agent("Agent1", X_POS)
        self.agent2 = Agent("Agent2", O_POS)

    def update_agent_policies(self):
        """
            Get the game state, and update the policies of each agent accordingly. 
        """
        winner, is_draw = get_winner(self.board)
        agent1_reward = agent2_reward = 0
        if is_draw:
            # if there's a draw, give a very low reward to agent1
            # we don't want agent1 to end games in a draw state
            agent1_reward = 0.05

            # agent2 gets a reward if he ends the game in a draw state,
            # because when agent2 manages to corner agent1 into a draw state,
            # it means that agent2 is still smarter than agent1, thus, we need to reward agent2 more
            agent2_reward = 0.6

        # agent1 won, give it a reward
        elif winner == self.agent1.symbol:
            agent1_reward = 1

        # agent2 won, give it a reward
        elif winner == self.agent2.symbol:
            agent2_reward = 1

        # update the policies (learning step)
        self.agent1.update_policy(agent1_reward)
        self.agent2.update_policy(agent2_reward)

    def reset(self):
        """
            Resets the board and the agents at the end of each round.
        """
        self.board = get_empty_board()
        self.agent1.reset()
        self.agent2.reset()

    def train(self, iterations: int):
        for iteration in range(iterations):
            # print the training progress every 5%
            if iteration % (iterations//10) == 0:
                print(f"Progress {iteration/iterations * 100:0.2f}%")

            # let agent1 do the first move
            agent1_turn = True

            # while there's no winner/draw in the board
            while True:
                # get the agent that should play in this iteration
                agent = self.agent1 if agent1_turn else self.agent2

                # switch the boolean vairable so the other agent plays in the next iteration
                agent1_turn = not agent1_turn

                # get the action that the agent shall perform on the current board
                agent_action = agent.get_next_action(self.board)
                i, j = agent_action

                # mark the move of the agent on the current board
                mark_move(i, j, self.board, agent.symbol)

                # add the current board state to the list of  states that the agent has seen so far
                agent.board_states.append(get_board_state(self.board))

                # check if there's a winner or if there's a draw
                winner, is_draw = get_winner(self.board)
                if (is_draw) or (winner != EMPTY_POS):
                    break
                
            self.update_agent_policies()
            self.reset()

def main():
    if "y" in input("Train a new agent?(y/n) ").lower():
        agent_trainer = AgentTrainer()
        iterations = int(input("Please enter the number of training iterations[40,000~100,0000]: "))

        print("Training the two agents... ")
        agent_trainer.train(iterations)
        print("Done!")

        # save the policy of the best agent to a json file
        agent_trainer.agent1.save_agent_policy("trained_agent_policy.json")

    # compete againt the bot
    if os.path.exists("trained_agent_policy.json"):
        print("Playing against the trained agent")
        play_against_agent("trained_agent_policy.json")
    
    else:
        print("Couldn't find any trained agents to play against! You have to train an agent first!")

if __name__ == "__main__":
    main()