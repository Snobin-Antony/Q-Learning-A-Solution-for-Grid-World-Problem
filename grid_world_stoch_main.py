"""
author : Snobin Antony
reference: https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff
created on: 09/03/2023
To run the code install the packages by, pip install numpy, pip install pandas, pip install matplotlib, pip install openpyxl
and run, python antony-s1.py 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BOARD_ROWS = 5
BOARD_COLS = 5
WIN_STATE = (4, 4)
JUMP_STATE = (3, 3)
START_STATE = (1, 0)
BLACK_GRID = [(3,2),(2,2),(2,3),(2,4)]
DETERMINISTIC = False


class Grid_State:
    # initialize variiables and the grid board
    def __init__(self, state=START_STATE):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    # reward function for win state, jump state and all other states 
    def getReward(self,special_flag):
        if self.state == WIN_STATE and special_flag =="nojump":
            return 10
        elif self.state == WIN_STATE and special_flag =="jump":
            return 15
        else:
            return -1

    def isEndFunc(self):
        if (self.state == WIN_STATE):
            self.isEnd = True

    def epsilon_greedy_prob(self, action):
        if action == "north":
            return np.random.choice(["north", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "south":
            return np.random.choice(["south", "west", "east"], p=[0.8, 0.1, 0.1])
        if action == "west":
            return np.random.choice(["west", "north", "south"], p=[0.8, 0.1, 0.1])
        if action == "east":
            return np.random.choice(["east", "north", "south"], p=[0.8, 0.1, 0.1])

    def next_cell_pos(self, action):
        """
        action: north, south, west, east
        --------------------------------
          | 0 | 1| 2| 3| 4|
        0 |
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
            action = self.epsilon_greedy_prob(action)
            self.determine = True
            nxtState = self.next_cell_pos(action)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):
                if nxtState not in BLACK_GRID:
                    return nxtState
                # to jump the agent from (1,3) to (3,3) that means action is south and nxtState is (2,3)
                if nxtState == (2,3) and action == "south":
                    self.state = JUMP_STATE
        return self.state

class Grid_Agent:
    # initialise the actions, states, Q-values and optimization paramteres 
    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["north", "south", "west", "east"]
        self.State = Grid_State()
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
                    self.Q_values[(i, j)][a] = -1  # Q value is a dict of dict and it initialised with -1 so all actions will get -1 reward

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = -1
        action = ''

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        return action

    def takeAction(self, action):
        position = self.State.next_cell_pos(action)
        # update State
        return Grid_State(state=position)

    def reset(self):
        self.states = []
        self.State = Grid_State()
        self.isEnd = self.State.isEnd

    def train(self, rounds=100):
        i = 0
        self.cum_itr_list = [1]
        cum_itr = False
        while (i < rounds and cum_itr == False):
            print("-----------------------------------------------> episode ",i)
            # to the end of game back propagate reward
            print(self.Q_values[self.State.state])
            if self.State.isEnd:
                # back propagate
                if [(1, 3), "south"] in (self.states):
                    #print(self.states)
                    reward = self.State.getReward("jump")
                    flag_reward = reward
                else:
                    reward = self.State.getReward("nojump")
                    flag_reward = reward
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                print("Game End Reward", reward)
                cum_reward = reward
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    ##print(current_q_value)
                    # updating reward 5 in the back propagation, which gained the agent when it jumped 
                    if s[0] == (1,3) and s[1] == "south":
                        reward = 5
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    else:
                        reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                    # Calculating cumulative reward
                    cum_reward += reward
                # Calculating average cumulative reward
                avg_reward = cum_reward/(len(self.states))
                self.cum_itr_list.append(avg_reward)
                cum_reward = 0
                
                # to stop the episode if avg cumulative reward greater than 10 in 30 consecutive episode. tried 5 instead 10
                if all(i >= 5 for i in self.cum_itr_list[-30:]) is True:
                    if flag_reward == 15:
                    # if cum_itr_list[-30:].count(15) == 30:
                        print(self.cum_itr_list[-30:])
                        cum_itr = True
                        break

                self.reset()
                i += 1

            else:
                action = self.chooseAction()
                # append trace
                self.states.append([(self.State.state), action])
                print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = self.takeAction(action)
                # mark is end
                self.State.isEndFunc()
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
        df.to_excel('Q_table.xlsx')
        print("q table created as xlsx file")

    def showGridboard_plot(self):
        fig, axs = plt.subplots(BOARD_ROWS, BOARD_COLS, figsize=(10, 10))
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                max_q_value = max(self.Q_values[(i, j)].values())
                color = plt.cm.RdYlBu(max_q_value / max(self.Q_values[(i, j)].values()))
                text_color = color if max_q_value < 0 else 'black'
                axs[i, j].text(0.5, 0.5, f"{max_q_value:.2f}", fontsize=10, ha='center', va='center',
                            color=text_color,zorder=2)
                if (i , j ) in BLACK_GRID:
                    axs[i, j].set_facecolor((0.0, 0.0, 0.0, 0.91))
                elif (i , j ) == WIN_STATE:
                    axs[i, j].set_facecolor((0.0, 0.0, 1.0,0.6))
                elif (i , j ) == JUMP_STATE:
                    axs[i, j].set_facecolor((0.0, 1.0, 0.0, 0.6))
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                axs[i, j].set_title(f"[{i },{j }]")
        plt.tight_layout()
        plt.savefig("gridout.png")
        plt.show()

if __name__ == "__main__":
    ag = Grid_Agent()
    print("initial Q-values ... \n")
    print(ag.Q_values)

    ag.train(1000)
    print("latest Q-values ... \n")
    print(ag.Q_values)
    ag.gen_qtable()
    ag.showGridboard()
    ag.showGridboard_plot()
    ag.plot_graph()