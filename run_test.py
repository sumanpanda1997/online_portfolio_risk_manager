import json
import os
from pathlib import Path

import random
import math
import numpy as np
import pandas as pd
random.seed(1)


ramdomizer_list=[]

for i in range(0,100):
    ramdomizer_list.append(random.random())

class EpsilonGreedy:
    def __init__(self, epsilon, pulls_per_arm, avg_vals):
        self.epsilon = epsilon  # probability of explore
        self.pulls_per_arm = pulls_per_arm  # number of pulls for each arms
        self.avg_vals = avg_vals  # average amount of reward we've gotten from each arms
        return

    def initialize(self, n_arms):
        self.pulls_per_arm = [0] * n_arms
        self.avg_vals = [0.0] * n_arms
        return

    def arm_selection(self):
        # For 1-epsilon probability the current average maximum valued arm will be chosen
        if random.random() > self.epsilon:
            return self.avg_vals.index(max(self.avg_vals))
        # A random other arm will be chosen
        else:
            return random.randrange(len(self.avg_vals))

    def update(self, chosen_arm, reward):
        avg_val = self.avg_vals[chosen_arm]
        self.pulls_per_arm[chosen_arm] += 1
        pulls = self.pulls_per_arm[chosen_arm]
        updated_val = (((pulls - 1) * avg_val) + reward) / float(pulls)
        self.avg_vals[chosen_arm] = updated_val
        return


class StockHistoryArm:
    def __init__(self, gains):
        self.gains = gains

    def draw(self, ind):
        return self.gains[ind]



def ftrl_output_formatted(ftrl_model_set, stock_set, number_of_entries):
    iterator = [0.0] * number_of_entries
    chosen_stocks = [0.0] * number_of_entries
    rewards = [0.0] * number_of_entries
    cumulative_rewards = [0.0] * number_of_entries

    possible_rewards = [0.0] * len(stock_set)
    losses = [0.0] * len(stock_set)
    probabilities = [0.0] * len(stock_set)

    i = 0
    while i < number_of_entries:
        iterator[i] = i + 1
        j = 0
        while j < len(stock_set):
            
            possible_rewards[j] = stock_set[j][i]
            probabilities[j] = ftrl_model_set[j].predict()
            losses[j] += FollowTheRegularizedLeader.log_loss(stock_set[j][i], probabilities[j])
            ftrl_model_set[j].update(probabilities[j], stock_set[j][i])
            j += 1
        index_of_maximum_probability = probabilities.index(max(probabilities))
        chosen_stocks[i] = index_of_maximum_probability
        rewards[i] = possible_rewards[index_of_maximum_probability]

        if i == 0:
            cumulative_rewards[i] = rewards[i]
        else:
            cumulative_rewards[i] = cumulative_rewards[i - 1] + rewards[i] + (cumulative_rewards[i - 1] * rewards[i])
        i += 1

    return [iterator, chosen_stocks, rewards, cumulative_rewards]


def ftrl_exp_output_formatted(ftrl_model_set, stock_set, number_of_entries):
    iterator = [0.0] * number_of_entries
    chosen_stocks = [0.0] * number_of_entries
    rewards = [0.0] * number_of_entries
    cumulative_rewards = [0.0] * number_of_entries

    possible_rewards = [0.0] * len(stock_set)
    losses = [0.0] * len(stock_set)
    probabilities = [0.0] * len(stock_set)
    cum_probablities = [0.0] * len(stock_set)


    i = 0
    while i < number_of_entries:
        iterator[i] = i + 1
        j = 0
        while j < len(stock_set):
            possible_rewards[j] = stock_set[j][i]
            probabilities[j] = ftrl_model_set[j].predict()
            losses[j] += FollowTheRegularizedLeader.log_loss(stock_set[j][i], probabilities[j])
            cum_probablities[j]=cum_probablities[j]*0.75+stock_set[j][i]*0.25
            ftrl_model_set[j].update(probabilities[j], cum_probablities[j])
            j += 1
        index_of_maximum_probability = probabilities.index(max(probabilities))
        chosen_stocks[i] = index_of_maximum_probability
        rewards[i] = possible_rewards[index_of_maximum_probability]

        if i == 0:
            cumulative_rewards[i] = rewards[i]
        else:
            cumulative_rewards[i] = cumulative_rewards[i - 1] + rewards[i] + (cumulative_rewards[i - 1] * rewards[i])
        i += 1

    return [iterator, chosen_stocks, rewards, cumulative_rewards]

def ftrl_moving_output_formatted(ftrl_model_set, stock_set, number_of_entries):
    iterator = [0.0] * number_of_entries
    chosen_stocks = [0.0] * number_of_entries
    rewards = [0.0] * number_of_entries
    cumulative_rewards = [0.0] * number_of_entries

    possible_rewards = [0.0] * len(stock_set)
    losses = [0.0] * len(stock_set)
    probabilities = [0.0] * len(stock_set)

    probabilities_counter=[[] for i in range(len(stock_set))]
    print(probabilities_counter)

    

    i = 0
    while i < number_of_entries:
        iterator[i] = i + 1
        j = 0
        while j < len(stock_set):
            possible_rewards[j] = stock_set[j][i]
            probabilities[j] = ftrl_model_set[j].predict()
            probabilities_counter[j].append(probabilities[j])
            losses[j] += FollowTheRegularizedLeader.log_loss(stock_set[j][i], probabilities[j])
            

            pred_norm=probabilities[j]
            if i-2>=0:
                pred_norm=(probabilities_counter[j][i]*2+probabilities_counter[j][i-1]*0.5+probabilities_counter[j][i-2]*0.5)/3

            ftrl_model_set[j].update(pred_norm, stock_set[j][i])
            j += 1
        index_of_maximum_probability = probabilities.index(max(probabilities))
        chosen_stocks[i] = index_of_maximum_probability
        rewards[i] = possible_rewards[index_of_maximum_probability]

        if i == 0:
            cumulative_rewards[i] = rewards[i]
        else:
            cumulative_rewards[i] = cumulative_rewards[i - 1] + rewards[i] + (cumulative_rewards[i - 1] * rewards[i])
        i += 1

    return [iterator, chosen_stocks, rewards, cumulative_rewards]

class StockInfo:
    def __init__(self, ticker, entries):
        self.ticker = ticker
        self.entries = entries
        self.filename = "data/JSON/" + self.ticker + "_stock_data.json"
        self.total_cumulative_gain_string = ""
        self.stock_time_interval = []
        self.gain_list = []
        self.cumulative_gain_list = []
        self.calculate_gains()
        self.calculate_cumulative_gain()

    def find_dates(self):
        try:
            print(self.filename)
            print(Path(self.filename))
            print(os.listdir())
            my_abs_path = Path(self.filename).resolve(strict=True)
            

        except FileNotFoundError:
            print("File does not exist - calling API")
            #JSON_REQUEST(self.ticker, "full")
        else:
            with open(self.filename) as json_data:
                handle = json.load(json_data)
                i = 0
                for data in handle["Time Series (Daily)"]:
                    if i < self.entries:
                        self.stock_time_interval.append(data)
                    i += 1
                self.stock_time_interval.reverse()

    def calculate_gains(self):
        i = 0
        previous = 0
        self.find_dates()
        with open(self.filename) as json_data:
            handle = json.load(json_data)
            for day in self.stock_time_interval:
                current = handle["Time Series (Daily)"][day]["4. close"]
                if i == 0:
                    previous = handle["Time Series (Daily)"][day]["4. close"]
                else:
                    gains = ((float(current) - float(previous)) / float(previous))
                    previous = handle["Time Series (Daily)"][day]["4. close"]
                    self.gain_list.append(gains)
                i += 1

    def calculate_cumulative_gain(self):
        i = 0
        cumulative_gain = 0
        for gain in self.gain_list:
            current = gain
            if i == 0:
                previous = gain
                cumulative_gain += gain
            else:
                cumulative_gain += current + (current * previous)
                previous = cumulative_gain
            i += 1
            self.cumulative_gain_list.append(cumulative_gain)
        self.total_cumulative_gain_string = str(round(self.cumulative_gain_list[-1] * 100, 4)) + '%'

class StockHistoryArm:
    def __init__(self, gains):
        self.gains = gains

    def draw(self, ind):
        return self.gains[ind]


class FollowTheRegularizedLeader(object):
    def __init__(self, alpha, beta, l1, l2, features, reg_func, indices,stock_no):
        self.alpha = alpha   #learning rate
        self.beta = beta   #Gradient smoothing parameter
        self.l1 = l1  #l1 regularizer
        self.l2 = l2 #l2 regularizer
        self.reg_func = reg_func #Sigmoid or Relu
        self.indices = [0]
        self.count = 0 #iter counts
        for index in indices: 
            self.indices.append(index)
        self.sum_of_gradients_squared = [0.] * features #squared gradients sum
        self.weights = {} #updated weights
        self.temp_weights = [] #weights temp holder
        i = 0
        while i < features:
            self.temp_weights.append(ramdomizer_list[stock_no])#random initializing of weights
            i += 1

    def update(self, probability, result):
        gradient = probability - result
        gradient_squared = math.pow(gradient, 2)
        self.count += 1
        if self.reg_func is "RDA":
            for i in self.indices:
                sigma = (math.sqrt(self.sum_of_gradients_squared[i] + gradient_squared)) / (self.alpha * self.count)
                self.temp_weights[i] += -(sigma * self.weights[i]) + gradient
                self.sum_of_gradients_squared[i] += gradient_squared
        elif self.reg_func is "OPG":
            for i in self.indices:
                sigma = (math.sqrt(self.sum_of_gradients_squared[i] + gradient_squared) - math.sqrt(self.sum_of_gradients_squared[i])) / self.alpha
                self.temp_weights[i] += -(sigma * self.weights[i]) + gradient
                self.sum_of_gradients_squared[i] += gradient_squared
        elif self.reg_func is "SGD":
            for i in self.indices:
                self.temp_weights[i] += self.alpha * gradient
                
            
    

    def predict(self):
        weights = {}
        function = "Sigmoid"
        w_inner_x = float(0)
        
        for i in self.indices:
            sign = float(np.sign(self.temp_weights))
            if self.reg_func=='SGD':
                weights[i]=self.temp_weights[i]
            elif sign * self.temp_weights[i] <= self.l1:
                # w[i] vanishes due to L1 regularization
                weights[i] = float(0)
            else:
                # apply prediction time L1, L2 regularization to weights and get temp_weights
                weights[i] = (sign * self.l1 - self.temp_weights[i]) / ((self.beta + math.sqrt(self.sum_of_gradients_squared[i])) / self.alpha + self.l2)
            w_inner_x += weights[i]
                
        self.weights = weights



        if function is "Sigmoid":
            probability = float(1) / (float(1) + math.exp(-max(min(float(w_inner_x), float(100)), float(-100))))
        elif function is "ReLU":
            probability = max(float(0), max(min(float(w_inner_x), float(100)), float(-100)))

        return probability

    @staticmethod
    def log_loss(true_label, predicted, eps=1e-15):
        p = np.clip(predicted, eps, 1 - eps)
        if true_label > 0:
            return -math.log(p)
        else:
            return -math.log(1 - p)


def automate_result(number_of_stocks,number_of_entries,stock_set,ftrl_func,params):

    ftrl_instances=[]
    for i in range(0,number_of_stocks):
        ftrl_instances.append(FollowTheRegularizedLeader(params[0],params[1],params[2],params[3],params[4],params[5],params[6],i))
    

    results= ftrl_func(ftrl_instances, stock_set, number_of_entries - 1)
    
    for i in ftrl_instances:
        del i
    del ftrl_instances

    return results


number_of_entries = 900

apple = StockInfo("AAPL", number_of_entries)
print(apple.ticker + ": " + apple.total_cumulative_gain_string)

microsoft = StockInfo("MSFT", number_of_entries)
print(microsoft.ticker + ": " + microsoft.total_cumulative_gain_string)

intel = StockInfo("INTC", number_of_entries)
print(intel.ticker + ": " + intel.total_cumulative_gain_string)

google = StockInfo("GOOGL", number_of_entries)
print(google.ticker + ": " + google.total_cumulative_gain_string)

amazon = StockInfo("AMZN", number_of_entries)
print(amazon.ticker + ": " + amazon.total_cumulative_gain_string)

twitter = StockInfo("TWTR", number_of_entries)
print(twitter.ticker + ": " + twitter.total_cumulative_gain_string)

tesla = StockInfo("TSLA", number_of_entries)
print(tesla.ticker + ": " + tesla.total_cumulative_gain_string)

fitbit = StockInfo("FIT", number_of_entries)
print(fitbit.ticker + ": " + fitbit.total_cumulative_gain_string)

altaba = StockInfo("AABA", number_of_entries)
print(altaba.ticker + ": " + altaba.total_cumulative_gain_string)

general_electric = StockInfo("GE", number_of_entries)
print(general_electric.ticker + ": " + general_electric.total_cumulative_gain_string)


qualcomm = StockInfo("QCOM", number_of_entries)
print(qualcomm.ticker + ": " + qualcomm.total_cumulative_gain_string)

sony = StockInfo("SNE", number_of_entries)
print(sony.ticker + ": " + sony.total_cumulative_gain_string)

cisco = StockInfo("CSCO", number_of_entries)
print(cisco.ticker + ": " + cisco.total_cumulative_gain_string)

activision = StockInfo("ATVI", number_of_entries)
print(activision.ticker + ": " + activision.total_cumulative_gain_string)

xilinx = StockInfo("XLNX", number_of_entries)
print(xilinx.ticker + ": " + xilinx.total_cumulative_gain_string)


stock_set_increasing = [apple.gain_list, google.gain_list, amazon.gain_list, intel.gain_list, microsoft.gain_list]
stock_set_big = [apple.gain_list, google.gain_list, amazon.gain_list, intel.gain_list, microsoft.gain_list, twitter.gain_list, altaba.gain_list, general_electric.gain_list, fitbit.gain_list, tesla.gain_list, qualcomm.gain_list, sony.gain_list, cisco.gain_list, activision.gain_list, xilinx.gain_list]

###reults for increasing set #####


result_df=pd.DataFrame(columns=['DAY_NO','FTRL_OPG_CUMILATIVE_GAIN','FTRL_RDA_CUMILATIVE_GAIN',
                                'FTL_CUMILATIVE_GAIN','FTRL_EXP_RDA_CUMILATIVE_GAIN',
                                'FTRL_MOVING_RDA_CUMILATIVE_GAIN','SGD_CUMILATIVE_GAIN'])


results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(2, 0.05, 1.5, 0.05, 1, "OPG", [0]))
result_df['DAY_NO']=results[0]
result_df['FTRL_OPG_CUMILATIVE_GAIN']=results[3]

results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(1.5, 0.1, 0.01, 0.1, 1, "RDA", [0]))
result_df['FTRL_RDA_CUMILATIVE_GAIN']=results[3]
del results

results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(0.05, 1.0, 0.0, 0.0, 1, "OPG", [0]))
result_df['FTL_CUMILATIVE_GAIN']=results[3]
del results

results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_exp_output_formatted,(0.05, 1.0, 1.0, 1.0, 1, "RDA", [0]))
result_df['FTRL_EXP_RDA_CUMILATIVE_GAIN']=results[3]
del results

results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_moving_output_formatted,(0.05, 1.0, 1.0, 1.0, 1, "RDA", [0]))
result_df['FTRL_MOVING_RDA_CUMILATIVE_GAIN']=results[3]
del results


results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(0.05, 1.0, 0.0, 0.0, 1, "SGD", [0]))
result_df['SGD_CUMILATIVE_GAIN']=results[3]
del results


print(result_df.columns)
result_df.to_csv("data/only_increasing_stocks_results_final.csv")

grid_param={
    'alpha':[0.05,0.1,1,1.5,2],
    'beta':[0.05,0.1,0.5,1,1.5,2],
    'l1':[0.05,0.1,0.5,1,1.5,2],
    'l2':[0.05,0.1,0.5,1,1.5,2]
}


columns_vector=['Day_No','Alpha=0.01','Aplha=0.02','Alpha=0.05','Alpha=0.1','Alpha=0.15','Alpha=0.25','Aplha=0.5','Alpha=0.75','Alpha=1']
results_df=pd.DataFrame(columns=columns_vector)
learning_rate=[0.01,0.02,0.05,0.10,0.15,0.25,0.5,0.75,1]
i=1

for eta in learning_rate:
    results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(eta, 1.0, 0.0, 0.0, 1, "SGD", [0]))
    results_df['DAY_NO']=results[0]
    results_df[columns_vector[i]]=results[3]   
    i=i+1

results_df.to_csv('data/only_increasing_sgd_learning_rate.csv')



grid_param={
    'alpha':[0.05,0.1,1,1.5,2],
    'beta':[0.05,0.1,0.5,1,1.5,2],
    'l1':[0.05,0.1,0.5,1,1.5,2],
    'l2':[0.05,0.1,0.5,1,1.5,2]
}


paramter_tuning_result=pd.DataFrame(columns=['alpha','beta','l1','l2','cum_gain'])

for a in grid_param['alpha']:
    for b in grid_param['beta']:
        for l1 in grid_param['l1']:
            for l2 in grid_param['l2']:
                results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(a,b,l1,l2,1,"RDA", [0]))
                paramter_tuning_result.loc[len(paramter_tuning_result)]=[a,b,l1,l2,results[3][-1]]



paramter_tuning_result.to_csv('data/only_increasing_param_tuning_FTRL_RDA.csv')


paramter_tuning_result=pd.DataFrame(columns=['alpha','beta','l1','l2','cum_gain'])

for a in grid_param['alpha']:
    for b in grid_param['beta']:
        for l1 in grid_param['l1']:
            for l2 in grid_param['l2']:
                results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_output_formatted,(a,b,l1,l2,1,"OPG", [0]))
                paramter_tuning_result.loc[len(paramter_tuning_result)]=[a,b,l1,l2,results[3][-1]]



paramter_tuning_result.to_csv('data/only_increasing_param_tuning_FTRL_OPG.csv')



paramter_tuning_result=pd.DataFrame(columns=['alpha','beta','l1','l2','cum_gain'])

for a in grid_param['alpha']:
    for b in grid_param['beta']:
        for l1 in grid_param['l1']:
            for l2 in grid_param['l2']:
                results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_exp_output_formatted,(a,b,l1,l2,1,"OPG", [0]))
                paramter_tuning_result.loc[len(paramter_tuning_result)]=[a,b,l1,l2,results[3][-1]]



paramter_tuning_result.to_csv('data/only_increasing_param_tuning_FTRL_OPG_exp_smoothen.csv')


paramter_tuning_result=pd.DataFrame(columns=['alpha','beta','l1','l2','cum_gain'])

for a in grid_param['alpha']:
    for b in grid_param['beta']:
        for l1 in grid_param['l1']:
            for l2 in grid_param['l2']:
                results=automate_result(len(stock_set_increasing),number_of_entries,stock_set_increasing,ftrl_moving_output_formatted,(a,b,l1,l2,1,"OPG", [0]))
                paramter_tuning_result.loc[len(paramter_tuning_result)]=[a,b,l1,l2,results[3][-1]]



paramter_tuning_result.to_csv('data/only_increasing_param_tuning_FTRL_OPG_moving_avg.csv')





###reults for big set #####
result_df=pd.DataFrame(columns=['DAY_NO','FTRL_OPG_CUMILATIVE_GAIN','FTRL_RDA_CUMILATIVE_GAIN',
                                'FTL_CUMILATIVE_GAIN','FTRL_EXP_RDA_CUMILATIVE_GAIN',
                                'FTRL_MOVING_RDA_CUMILATIVE_GAIN','SGD_CUMILATIVE_GAIN'])

			

results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(2, 0.05, 1.5, 0.05, 1, "OPG", [0]))
result_df['DAY_NO']=results[0]
result_df['FTRL_OPG_CUMILATIVE_GAIN']=results[3]


results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(1.5, 0.1, 0.01, 0.1, 1, "RDA", [0]))
result_df['FTRL_RDA_CUMILATIVE_GAIN']=results[3]
del results

results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(0.05, 1.0, 0.0, 0.0, 1, "OPG", [0]))
result_df['FTL_CUMILATIVE_GAIN']=results[3]
del results

results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_exp_output_formatted,(0.05, 1.0, 1.0, 1.0, 1, "RDA", [0]))
result_df['FTRL_EXP_RDA_CUMILATIVE_GAIN']=results[3]
del results

results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_moving_output_formatted,(0.05, 1.0, 1.0, 1.0, 1, "RDA", [0]))
result_df['FTRL_MOVING_RDA_CUMILATIVE_GAIN']=results[3]
del results


results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(0.05, 1.0, 0.0, 0.0, 1, "SGD", [0]))
result_df['SGD_CUMILATIVE_GAIN']=results[3]
del results


print(result_df.columns)
result_df.to_csv("data/big_stocks_results_final.csv")



columns_vector=['Day_No','Alpha=0.01','Aplha=0.02','Alpha=0.05','Alpha=0.1','Alpha=0.15','Alpha=0.25','Aplha=0.5','Alpha=0.75','Alpha=1']
results_df=pd.DataFrame(columns=columns_vector)
learning_rate=[0.01,0.02,0.05,0.10,0.15,0.25,0.5,0.75,1]
i=1

for eta in learning_rate:
    results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(eta, 1.0, 0.0, 0.0, 1, "SGD", [0]))
    results_df['DAY_NO']=results[0]
    results_df[columns_vector[i]]=results[3]   
    i=i+1

results_df.to_csv('data/big_sgd_learning_rate.csv')



paramter_tuning_result=pd.DataFrame(columns=['alpha','beta','l1','l2','cum_gain'])

for a in grid_param['alpha']:
    for b in grid_param['beta']:
        for l1 in grid_param['l1']:
            for l2 in grid_param['l2']:
                results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(a,b,l1,l2,1,"RDA", [0]))
                paramter_tuning_result.loc[len(paramter_tuning_result)]=[a,b,l1,l2,results[3][-1]]



paramter_tuning_result.to_csv('data/big_param_tuning_FTRL_RDA.csv')


paramter_tuning_result=pd.DataFrame(columns=['alpha','beta','l1','l2','cum_gain'])

for a in grid_param['alpha']:
    for b in grid_param['beta']:
        for l1 in grid_param['l1']:
            for l2 in grid_param['l2']:
                results=automate_result(len(stock_set_big),number_of_entries,stock_set_big,ftrl_output_formatted,(a,b,l1,l2,1,"OPG", [0]))
                paramter_tuning_result.loc[len(paramter_tuning_result)]=[a,b,l1,l2,results[3][-1]]



paramter_tuning_result.to_csv('data/big_param_tuning_FTRL_OPG.csv')
