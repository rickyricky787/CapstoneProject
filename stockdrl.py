#Samuel Khong
#CS 499

import csv
import numpy as np
import datetime,time
import random

BUY = 0
HOLD = 1
SELL = 2

epsilon = 0.1

actions_dict = {
BUY:'buy',
HOLD: 'hold',
SELL: 'sell',
}

amc = []
with open("AMC_pred.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        amc.append(row)

class Agent:
    def __init__(self,stock):
        self.stock = stock
        self.date = 1
        self.start = 1000.0
        self.portfolio = 1000.0
        self.shares = 0
        self.buypower = 1000.0

    def move(self, action):
        if self.date > len(self.stock)-1:
            if(self.portfolio > self.start*1.25):
                self.state = "win"
            else:
                self.state = "lose"
            return self.state, self.portfolio
        if(not self.validate(action)):
            self.state = "blocked"
        elif self.validate(action):
            if action == BUY:
                maxbuy = int(self.buypower/float(self.stock[self.date][1]))
                if(maxbuy == 1):
                    self.buypower = self.buypower - float(self.stock[self.date][1])
                    self.shares += 1
                else:
                    amount = int(random.randrange(1,maxbuy))
                    self.buypower = self.buypower - float(self.stock[self.date][1])*amount
                    self.shares += amount
            if action == SELL:
                self.buypower = self.buypower + float(self.stock[self.date][1])*float(self.shares)
                self.shares = 0
        else:
            self.state = "invalid"

        if(self.portfolio < self.start * 0.75):
            self.state = "lose"
        elif(self.portfolio > self.start * 1.25):
            self.state = "win"
        else:
            self.state = "valid"

        self.update_portfolio()
            
        return self.state, self.portfolio
            
    def update_portfolio(self):
        self.date += 1
        self.portfolio = float(float(self.shares) * float(self.stock[self.date][1])) + self.buypower


    def validate(self, action):
        if self.date == len(self.stock)-1:
            return False
        
        if action == 0:
            if self.buypower < float(self.stock[self.date][1]):
                return False

        if action == 2:
            if self.shares < 1:
                return False


        return True

    def check_rewards(self, action):
        if (action == 0):
            if(self.stock[self.date][1] > self.stock[self.date+1][1]):
                return -1
            else:
                return -.5
        if (action == 1):
            if(self.stock[self.date][1] > self.stock[self.date+1][1]):
               return -.1
            else:
                if(self.shares > 0):
                    return -.5
                else:
                    return -.51
        if(action == 2):
            if(self.stock[self.date][1] > self.stock[self.date-1][1]):
                return 1
            else:
                return -1
        return-10

    def reset(self):
        self.date = 1
        self.portfolio = 1000.0
        self.shares = 0
        self.buypower = 1000.0

    

class Qtable():
    def __init__(self, size):
        self.size = size
        self.qtable = np.zeros((3,size),float)
		    
    def predict(self, agent):
        best_moves = []
        mqscore = -100.0
        #print(self.qtable)
        #finds highest q value
        for action in range(len(self.qtable)):
            if(agent.validate(action)):
                if self.qtable[action,agent.date] > mqscore:
                    mqscore = self.qtable[action,agent.date]

        for action in range(len(self.qtable)):
            #print(self.qtable[action,maze.rat.row,maze.rat.col])
            if self.qtable[action,agent.date] == mqscore:
                best_moves.append(action)
        return best_moves

    def update(self,agent,actions):
        delta = []
        scores = []

        for moves in actions_dict:
            #print(scores)
            if(agent.validate(moves) and moves in actions):
                if moves == BUY:
                    scores.append(agent.check_rewards(moves))
                if moves == HOLD:
                    scores.append(agent.check_rewards(moves))
                if moves == SELL:
                    scores.append(agent.check_rewards(moves))
            else:
                scores.append(-10)
                
        counter = 0
        for action in actions_dict:

            for moves in actions_dict:
                if(agent.validate(moves) and moves in actions):
                    if moves == BUY:
                        delta.append(self.qtable[moves,agent.date+1])
                    if moves == HOLD:
                        delta.append(self.qtable[moves,agent.date+1])
                    if moves == SELL:
                        delta.append(self.qtable[moves,agent.date+1])

            if(agent.validate(action) and action in actions):
                #print(self.qtable[action,maze.rat.row,maze.rat.col])
                #print(scores)
                self.qtable[action,agent.date] = self.qtable[action,agent.date] + 0.1*(scores[counter] + 0.9 * max(delta) - self.qtable[action,agent.date])
                #print(self.qtable)
            else:
                self.qtable[action,agent.date] = self.qtable[action,agent.date] + (scores[counter] + 0.9* -1 - self.qtable[action,agent.date])


            counter+=1
            delta = []
            #print("action: " + str(action), " row: " + str(maze.rat.row), ", col: " + str(maze.rat.col))
            #print(scores)
            #print(counter)
                

        del delta
        del scores
        print(self.qtable)

def qtrain():
    global epsilon
    n_epoch = 1000
    max_memory = 100
    #data_size = 50
    #weights_file = ""
    start_time = datetime.datetime.now()

    #if not weights_file == "":
        #print("weights loading")

    amcagent = Agent(amc)
    qtable = Qtable(len(amc))
    win_history = []
    win_rate = 0.0
    imctr = 1
    optimal_count = 0
    final_portfolio = []

    for epoch in range(n_epoch):
        loss = 0.0
        #qmaze.reset()
        amcagent.reset()
        game_over = False
        
        #reset_screen(MAZE3)

        n_episodes = 0
        while not game_over:
            valid_moves = [moves for moves in actions_dict if amcagent.validate(moves)]
            if not valid_moves: break
            #prev_board = qmaze.board
            qtable.update(amcagent,valid_moves)

            rando = np.random.rand()
            if rando < epsilon:
                rando = np.random.rand()
                if(rando < 0.2):
                    action = 1
                else:
                    action = 0
                #action = random.choice(valid_moves)
            else:
                best_moves = qtable.predict(amcagent)
                action = random.choice(best_moves)

            status,reward = amcagent.move(action)
            #update_screen(qmaze)
            if(status == "win"):
                win_history.append(1)
                game_over = True
            elif(status == "lose"):
                win_history.append(0)
                game_over = True
            else:
                game_over = False


            if len(win_history) > 0:
                win_rate = sum(win_history) / len(win_history)

            dt = datetime.datetime.now() - start_time
            t = format_time(dt.total_seconds())
            template = "Epoch: {:03d}/{:d} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {} | portfolio = {:f}"
            print(template.format(epoch, n_epoch-1, n_episodes, sum(win_history), win_rate, t, reward))
            if win_rate > 0.9 : epsilon = 0.01
            n_episodes+=1
            #time.sleep(1)
        final_portfolio.append(reward)
        print(final_portfolio)
                

def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

qtrain()
