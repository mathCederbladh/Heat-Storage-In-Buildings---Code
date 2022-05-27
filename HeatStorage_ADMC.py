# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:43:39 2022

@author: August Dahlberg & Mathilda Cederblad
"""
# Import Libaries
import numpy as np
import time

# Import functions
from HeatStorageFun_ADMC import importdata, DDMcurvefit, TCM, optimizerPeakshaving, Simulation, optimizerSetTin


# Timer
start_time = time.time()
# import data

# Conditions
train = 408 # training amount
prediction_horizon = 48 # Prediction horizon
temp_set = 22 + 273.15 # Set temperature
temp_variation = 0.5 # allowed temperature variations 
mode = 0 # mode = 0 -> peakshaving, mode = 1 -> simulation based on Tin trend

filename = 
data_input = importdata(filename, mode)

#%%
end = len(data_input[:,0])# end time
start = train + prediction_horizon # start time
maxerror = np.zeros([end-start,])
meanerror = np.zeros([end-start,])
for i in range(start, end):
    # Training data
    train_inputs = data_input[i-train:i,:] # Training inputs
    if i == start: # determine R, Qpassive and C, only done for the first time step. 
        # Initial guess
        
        # convert the hole DDM to a function as mathilda did with TCM R, Qpassive, r2? = DDM(inputs, R0, Qpassive0)
        R0 = 1000 # initial guess heat loss rate 
        Qpassive0 = 20000 # initail guess passive heating
        R, Qpassive = DDMcurvefit(train_inputs, R0, Qpassive0)

        train_inputsTCM = np.column_stack((train_inputs, data_input[i-train-1:i-1, 3])) # add Ti-1 as a vector
        tau = TCM(train_inputsTCM, train, 1000, 100000) # tau is determined based on TCM
        C = tau*R

    test_inputs = data_input[i-prediction_horizon:i+1, :] 
    if mode == 0:
        if test_inputs[0,3] > temp_set + temp_variation: # figure out how to do this in a better way...
            Qoptimized, Toptimized = optimizerSetTin(test_inputs, R, Qpassive, C, temp_set)
        else: 
            Qoptimized, Toptimized = optimizerPeakshaving(test_inputs, R, Qpassive, C, temp_variation, temp_set)
        data_input[i-prediction_horizon+1,3] = Toptimized # replace with the optimized T
    elif mode == 1:
        Qoptimized = Simulation(test_inputs, R, Qpassive, C)
    data_input[i-prediction_horizon+1,1] = Qoptimized # replace with the optimized Q
end_time = time.time()
print ("Time elapsed:", end_time - start_time)