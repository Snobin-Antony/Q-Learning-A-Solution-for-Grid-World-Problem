import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
START = (1, 0)
DETERMINISTIC = False
JUMP_STATE = (3, 3)
JUMP_START_STATE = (1, 3)

BLACK_GRID = [(3,2),(2,2),(2,3),(2,4)]


class State:
    def __init__(self, state=START):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self, prevstate):
        if self.state == WIN_STATE:
            return 10
        elif self.state == JUMP_STATE and prevstate == JUMP_START_STATE:
            return 5
        else:
            return -1

    def isEndFunc(self,prevstate):
        if (self.state == WIN_STATE) :
        # if (prevstate == WIN_STATE) :
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "north":
            return np.random.choice(["north", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "south":
            return np.random.choice(["south", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "west":
            return np.random.choice(["west", "north", "south"], p=[0.8, 0.1, 0.1])
        if action == "east":
            return np.random.choice(["east", "north", "south"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        """
        action: north, south, west, east
        -------------
        0 | 1 | 2| 3|4
        1 |
        2 |
        3 |
        4 |
        return next position on board
        """
        if self.determine:
            if action == "north":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "south":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "west":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):
                if nxtState not in BLACK_GRID:
                    return nxtState
                # to jump the agent from (1,3) to (3,3) that means action is south and nxtState is (2,3)
                if nxtState == (2,3) and action == "south":
                    self.state = JUMP_STATE
        return self.state


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["north", "south", "west", "east"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.2 # 1 # 0.9 # 0
        self.exp_rate = 0.3
        self.decay_gamma = 0.9 # 1 # 0.2

        # initial Q values
        self.Q_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = -1  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        # exploration
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.State.state
            action_dict = self.Q_values[current_position]
            action = max(action_dict, key=action_dict.get)
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd
    def updateQvalue(self, cur_state, action, next_state, reward):
        current_q_value = self.Q_values[cur_state][action]
        q_val_array = self.Q_values[next_state]
        max_q_value = max(q_val_array.values())
        # target_q_value = self.Q_values[next_state][action]
        current_q_value = current_q_value + self.lr*(reward + self.decay_gamma * max_q_value - current_q_value)
        ## SARSA
        # current_q_value = current_q_value + self.lr*(reward + self.decay_gamma * target_q_value - current_q_value)
        self.Q_values[cur_state][action] = round(current_q_value,3)


    def play(self, rounds=10):
        i = 0
        cum_reward = 0
        self.cum_itr_list = [1]
        cum_itr = False
        while (i < rounds and cum_itr == False):
            print("-----------------------------------------------> episode ",i)
            # checking if end state is reached or one episode is completed
            if self.State.isEnd:
                reward = self.State.giveReward(prevstate)
                cum_reward += reward
                # updating the the Q value for the current state and action
                for a in self.actions:
                    self.updateQvalue(self.State.state, a, self.State.state, reward)
                # self.updateQvalue(prevstate, action, self.State.state, reward)
                # Calculating average cumulative reward
                avg_reward = cum_reward/(len(self.states))
                self.cum_itr_list.append(avg_reward)
                print(f'average reward {avg_reward}')
                cum_reward = 0
                
                # to stop the episode if avg cumulative reward greater than 10 in 30 consecutive episode. tried 5 instead 10
                if all(i >= 1 for i in self.cum_itr_list[-30:]) is True:
                    # if flag_reward == 15:
                    # if cum_itr_list[-30:].count(15) == 30:
                    print(self.cum_itr_list[-30:])
                    cum_itr = True
                    break
                
                self.reset()
                i += 1
            else:
                # the present state of the environment
                prevstate = self.State.state
                # choosing the action randomly or using greedy policy
                action = self.chooseAction()

                print("current position {} action {}".format(prevstate, action))
                # as a result of action the enviroment is changed to next state
                self.State = self.takeAction(action)
                # the reward is given by the enviroment
                reward = self.State.giveReward(prevstate)
                cum_reward += reward
                # print(f'the reward {reward}')
                # the current state and reward are appended to the list of states in the episode
                self.states.append([(prevstate), reward])
                # updating the the Q value for the current state and action
                self.updateQvalue(prevstate, action, self.State.state, reward)

                # mark is end
                self.State.isEndFunc(prevstate)
                print("nxt state", self.State.state)
                self.isEnd = self.State.isEnd

    def plot_graph(self):
        # to plot the graph of episode vs avg_cumulative_reward
        plt.plot(self.cum_itr_list)
        # naming the x axis
        plt.xlabel('Episodes')
        # naming the y axis
        plt.ylabel('Average Reward')
        # giving a title to graph
        plt.title('Learning Performance graph')
        plt.show()

    # function for printing the grid with max(q-values)
    def showGridboard(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(max(self.Q_values[(i, j)].values())).ljust(6) + ' | '
                # out += str(self.Q_values[i, j]['up']).ljust(6) + ' | '
                # out += str(self.Q_values[(i, j)]).ljust(6) + ' | '
            print(out)
        print('----------------------------------------------')

    # to generate q-table with dataframe
    def gen_qtable(self):
        df = pd.DataFrame.from_dict(self.Q_values, orient = 'index')
        df.columns = [['Action', 'Action', 'Action', 'Action'], ['north', 'south', 'west', 'east']]  
        df = df.reset_index()
        df['level_0'] = df['level_0'].astype(str)
        df['level_1'] = df['level_1'].astype(str)
        df.insert(0, 'States', '('+df['level_0']+','+df['level_1']+')')
        df = df.drop(['level_0','level_1'], axis=1)
        print(df)
        # Write DataFrame to Excel file
        # pip install openpyxl
        # df.to_excel('Q_table.xlsx')
        # print("q table created as xlsx file")

if __name__ == "__main__":
    ag = Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)
    ag.showGridboard()

    ag.play(1000)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    ag.gen_qtable()
    ag.showGridboard()
    ag.plot_graph()

    # new_arr_reward = []
    # new_arr_episode = ag.episode_array[np.where(ag.episode_array != 0)]
    # i = 0

    # while i < len(new_arr_episode):
    #     new_arr_reward.append(ag.cum_reward_list[i])
    #     i += 1

    # #print(len(new_arr_reward))
    # #print(len(new_arr_episode))
    # if len(new_arr_episode) == len(new_arr_reward):
    #     plt.plot(new_arr_episode, new_arr_reward)
    #     plt.xlabel('Episode')
    #     plt.ylabel('Cumulative Reward')
    #     plt.title('Cumulative Reward Accross Episodes')
    #     plt.show()
    # else:
    #     pass