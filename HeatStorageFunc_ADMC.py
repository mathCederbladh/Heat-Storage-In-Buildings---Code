"""
Created on Fri May 27 09:43:39 2022

@author: August Dahlberg & Mathilda Cederblad
"""
import numpy as np
import pandas as pd
import scipy.optimize as so
from gekko import Gekko

def importdata(filename, mode):
    # Import data
    
    if mode == 0:
        dataset = pd.read_csv(filename)
        dataset['Q_supplyW'] = (dataset['T_supplyK']-dataset['T_returnK'])*dataset['Mass_flowkgs']*4180
        dataset['Time1h'] = dataset.index
        dataset['Time'] = pd.to_datetime(dataset['Time'])
        dataset.index = dataset['Time']
        # Devide data between inputs and output
        data_input = dataset[['Time1h', 'Q_supplyW', 'T_extK', 'T_in_avgK']]
        data_input = data_input.values
    
    elif mode == 1:
        data = pd.read_csv(filename)
        data['Time1h'] = data.index
        data['Time'] = pd.to_datetime(data['Time'])
        data.index = data['Time']
        
        Tin = data['T_in_avgK'].loc['2019-12-07 00':'2020-02-01 22']
    
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        result = seasonal_decompose(Tin, model='additive')
        trend = result.trend
        trend = pd.DataFrame(trend)
        trend.fillna(method='ffill')
        data['Q_supplyW'] = (data['T_supplyK']-data['T_returnK'])*data['Mass_flowkgs']*4180
        data_input = data[['Time1h', 'Q_supplyW', 'T_extK', 'T_in_avgK']].values
        data_input[381:1487-12, 3] = trend[381 - 120:1367-12]['trend']
        data_input = data_input[:1487-12,:]
    
    return data_input

def DDMcurvefit(inputs, R0, Qpassive0):
    p_opt, p_cov = so.curve_fit(f=DDM,
                        xdata=inputs,
                        ydata=inputs[:,1], 
                        p0=(R0, Qpassive0))
    R = p_opt[0]
    Qpassive = p_opt[1]
    """
    # rearrange the training data to by sequential order based on 
    # temperature difference between indoor and outdoor temperature
    test_r2[:,0] = train_inputs[:,1]
    test_r2[:,1] = train_inputs[:,3] - train_inputs[:,2]
    test_r2= test_r2[test_r2[:, 1].argsort()]
    
    # calculate r^2 to determine how good a and b fits
    fit = fitaxb(test_r2[:,1], R, Qpassive)
    r2[i-start-prediction_horizon] = r2_score(test_r2[:,0], fit)
    """   
    return R, Qpassive

def DDM(inputs, R, Qpassive):
    """

    Parameters
    ----------
    inputs : Indoor temperature and Outdoor temperature [K]
    R : Heat loss rate of the building [W/K]
        The heat loss rate represents how much heat is lost to the buildings surrondings for every degree of 
        difference between the indoor and outdoor temperature
    Qpassive : Passive heating of the building [W]
        The passive heating of the building represents the the heating from other sources then the district heating network, 
        this could include passive heating from occupanies and from appliencies

    Returns
    -------
    The external heat that is required for the building to hold a steady temperature
        DESCRIPTION.

    """
    Tin = inputs[:,3]
    Text = inputs[:,2]
    
    return -R*(Text-Tin) - Qpassive 

def FOTM(inputs, R, Qpassive, C):
    
    time = inputs[:, 0]
    Qsupply = inputs[:, 1]
    Text = inputs[:, 2]
    Tin = np.zeros(len(time))
    # Initial temperatures
    Tin[0] = inputs[0,3] 

    # Loop for calculating all temperatures
    for t in range(1, len(time)):
        dt = time[t] - time[t-1]
        Tin[t] = Tin[t-1] + dt *((R*(Text[t-1]-Tin[t-1]) + Qpassive)/C + Qsupply[t-1]/(C)) 
    return Tin

def TCM(inputs,trainLen, U0, C0):
    
    def Fopt(C):
        Tin=inputs[i,2]-(inputs[i,2]-inputs[i,4]+(inputs[i,1])/C[0])*np.exp(-C[0]/C[1])+(inputs[i,1])/C[0]
        
        return abs(Tin-inputs[i,3])
        
    res1 = np.zeros([trainLen,3])
    for i in range(0,trainLen):

        res=so.least_squares(Fopt, [U0, C0], ftol=1e-08)
        
        res1[i,0], res1[i,1], res1[i,2]=res.x[0], res.x[1], res.cost
        
    print('L = ' + str(np.mean(res1[:,0])))
    print('C = ' + str(np.mean(res1[:,1])))
    print('Tau = ' + str(np.mean(res1[:,1])/np.mean(res1[:,0])))
    
    return np.mean(res1[:,1])/np.mean(res1[:,0])

def optimizerPeakshaving(inputs, R, Qpassive, C, variation, set_temperature):
    """
    Parameters
    ----------
    inputs : numpy array
        Indoor temperature, external temperature, heat supplied for the prediction forecast.  
    R : float
        The buildings steady heat loss 
    Qpassive : float
        Heat supplied from other sources then the heat supplied from the DHN
    C : float
        Thermal capacity of the building

    Returns
    -------
    Q : numpy array
    Heat supplied array for the coming 48 hours
    T : numpy array
    Indoor temperature array for the coming 48 hours

    """
    m = Gekko(remote=False)
    #inputs = data_input[400:420+1,:]
    forecast = len(inputs[:,0])
    Text = inputs[:,2]
    p = np.zeros([forecast-1,])
    p[:] = 10
    p[1] = 1
    Qsupply = [m.Var(lb=2000, ub=50000) for i in range(forecast-1)]
    
    
    Tin = [m.Var(lb=set_temperature - variation, ub=set_temperature + variation) for i in range(forecast)]
    
    m.Equation(Tin[0] == inputs[0,3])
    m.Equation(Qsupply[0] == inputs[0,1])
    m.Equation(Tin[1] == Tin[0] + (R*(Text[0]-Tin[0]) + Qsupply[0] + Qpassive)/C)
    for i in range(1, forecast-1):
        m.Equation(Qsupply[i] == (Tin[i+1] - Tin[i])*C - R*(Text[i]-Tin[i]) - Qpassive)
    m.Obj(sum((Qsupply[i]-Qsupply[i-1])*(Qsupply[i]-Qsupply[i-1])*p[i]  for i in range(1,forecast-1)))

    m.options.IMODE=3 #Steady State Control problem
    m.options.SOLVER = 3
    m.solve(disp=True)
    """
    T = np.zeros([forecast-1,])
    Q = np.zeros([forecast-1,])
    for i in range(0,forecast-1):
        Q[i] = Qsupply[i].value[0]
        T[i] = Tin[i].value[0]
    """
    T = Tin[1].VALUE[0]
    Q = Qsupply[1].VALUE[0]
    return Q, T

def Simulation(inputs, R, Qpassive, C):
    """
    Parameters
    ----------
    inputs : numpy array
        Indoor temperature, external temperature, heat supplied for the prediction forecast.  
    R : float
        The buildings steady heat loss 
    Qpassive : float
        Heat supplied from other sources then the heat supplied from the DHN
    C : float
        Thermal capacity of the building

    Returns
    -------
    Q : numpy array
    Heat supplied array for the coming 48 hours
    T : numpy array
    Indoor temperature array for the coming 48 hours

    """
    m = Gekko(remote=False)
    #inputs = data_input[400:420+1,:]
    forecast = len(inputs[:,0])
    Text = inputs[:,2]
    Tin = inputs[:,3]


    Qsupply = [m.Var() for i in range(forecast-1)]

    
    m.Equation(Qsupply[0] == inputs[0,1])
    for i in range(1, forecast-1):
        m.Equation(Qsupply[i] == (Tin[i+1] - Tin[i])*C - R*(Text[i]-Tin[i]) - Qpassive)

    m.options.IMODE=1 #Steady State Control problem
    m.solve(disp=True)
    """
    Q = np.zeros([forecast-1,])
    for i in range(0,forecast-1):
        Q[i] = Qsupply[i].value[0]
        #T[i] = Tin[i].value[0]
    """
    Q = Qsupply[1].value[0]
    return Q

def optimizerSetTin(inputs, R, Qpassive, C, set_temperature):
    
    
    """
    Parameters
    ----------
    inputs : numpy array
        Indoor temperature, external temperature, heat supplied for the prediction forecast.  
    R : float
        The buildings steady heat loss 
    Qpassive : float
        Heat supplied from other sources then the heat supplied from the DHN
    C : float
        Thermal capacity of the building

    Returns
    -------
    Q : numpy array
    Heat supplied array for the coming 48 hours
    T : numpy array
    Indoor temperature array for the coming 48 hours

    """
    m = Gekko(remote=False)
    #inputs = data_input[400:420+1,:]
    forecast = len(inputs[:,0])
    Text = inputs[:,2]
    p = np.zeros([forecast-1,])
    p[:] = 10
    p[1] = 1
    Qsupply = [m.Var(lb=2000, ub=50000) for i in range(forecast-1)]
    
    
    Tin = [m.Var(lb=set_temperature - 3, ub=set_temperature + 3) for i in range(forecast)]
    
    m.Equation(Tin[0] == inputs[0,3])
    m.Equation(Qsupply[0] == inputs[0,1])
    for i in range(0, forecast-1):
        #m.Equation(Qsupply[i] == (Tin[i+1] - Tin[i])*C - R*(Text[i]-Tin[i]) - Qpassive)
        m.Equation(Tin[i+1] == Tin[i] + 1/C*(R*(Text[i]-Tin[i]) + Qsupply[i] + Qpassive))
    #m.Obj(sum((Tin[i]-set_temperature)**2  for i in range(1,forecast)))
    m.Obj(sum((Tin[i]-(set_temperature))**2*p[i]  for i in range(1,forecast-1)))
    
    m.options.IMODE=3 #Steady State Control problem
    #m.solver_options = ['linear_solver ma57']
    m.options.SOLVER = 3
    m.solve(disp=True)

    T = Tin[1].VALUE[0]
    Q = Qsupply[1].VALUE[0]
    return Q, T