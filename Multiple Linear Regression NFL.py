
# coding: utf-8

# In[ ]:

#***Single LINEAR REGRESSION***#


# In[157]:

import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import statsmodels.api as stats
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
get_ipython().magic('matplotlib inline')


# In[158]:

df1 = pd.read_excel("Master NFL File.xlsx")


# In[159]:

Temp_Season = df1[df1.Season < 2014]    
df = Temp_Season[['Win_Percentage', 'Offensive_Yards_Gained']]


# In[141]:

# Normaliztion
# df = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))


# In[142]:

df.head()


# In[160]:

df.corr()


# In[144]:

def computewinpercentage(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# In[145]:

df.insert(0, 'Ones', 1)


# In[146]:

cols = df.shape[1]
X = df.iloc[:,[0,2]]
y = df.iloc[:,[1]]


# In[147]:

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))


# In[148]:

X.shape, theta.shape, y.shape 


# In[149]:

computewinpercentage(X, y, theta)


# In[150]:

def gradientDescent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    win = np.zeros(iterations)
    
    for i in range(iterations):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        win[i] = computewinpercentage(X, y, theta)
        
    return theta, win


# In[151]:

alpha = 0.01
iterations = 1000


# In[152]:

g, win = gradientDescent(X, y, theta, alpha, iterations)
g


# In[153]:

computewinpercentage(X, y, g)


# In[161]:

result = sm.ols(formula="Win_Percentage ~ Offensive_Yards_Gained", data=df).fit()
print result.summary()


# In[155]:

x = np.linspace(df['Win_Percentage'].min(), df['Win_Percentage'].max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(df['Win_Percentage'], df.Offensive_Yards_Gained, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Win Percentage')
ax.set_ylabel('Offensive Yards')
ax.set_title('Win Percentage vs. Offensive Yards')


# In[156]:

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iterations), win, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Wins')
ax.set_title('Error vs. Training Epoch')


# In[ ]:

# Multiple Linear Regression Stats Package


# In[4]:

data2 = Temp_Season.drop(['Teams', 'Season','Away_Ties', 'Total_Ties', 'Home_Ties'], 1)


# In[12]:

df1 = pd.read_excel("Master NFL File (1).xlsx")
Temp_Season = df1[df1.Season < 2014] 
result_test = sm.ols(formula="Win_Percentage ~ Sacks_Allowed + Offensive_Yards_Gained + Defensive_Yards_Allowed + Points_For + Points_Against + Avg_Margin_of_Victory + Turnover_Differential + Opponent_Win_Percentage + Time_Of_Possession", data=Temp_Season).fit()
print result_test.summary()


# In[ ]:

# Multiple Linear Regression Stats Package Gradient Descent
data3 = data2[['Offensive_Yards_Gained', 'Points_Against', 'Win_Percentage']]
# Normalize
data3 = data3.apply(lambda x: (x - np.mean(x)) / (np.std(x)))
# Insert Ones
data3.insert(0, 'Ones', 1)
data3.head()
# Stats Regression Analysis
result = sm.ols(formula="Win_Percentage ~ Offensive_Yards_Gained + Points_Against", data=data3).fit()
print result.summary()
# set X (training data) and y (target variable)
cols = data3.shape[1]  
X2 = data3.iloc[:,0:cols-1]  
y2 = data3.iloc[:,cols-1:cols]
# convert to matrices and initialize theta
X2 = np.matrix(X2.values)  
y2 = np.matrix(y2.values)  
theta2 = np.matrix(np.array([0,0,0]))
# Type 1 error and Iterations
alpha = 0.01
iterations = 1000
# perform linear regression on the data set
g2, win2 = gradientDescent(X2, y2, theta2, alpha, iterations)
# get the cost (error) of the model
computewinpercentage(X2, y2, g2)
# Error
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iterations), win2, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Error')  
ax.set_title('Error vs. Training Epoch') 


Prediction_Data = pd.read_excel('Test_2016.xlsx')


def Super_Bowl_Odds(Current_Data):
    pass
    win_percentage=1.2395+(-0.0014*Current_Data.loc[i,'Sacks_Allowed'])+(-.00001363*Current_Data.loc[i,'Offensive_Yards_Gained'])+(.000005341*Current_Data.loc[i,'Defensive_Yards_Allowed'])+(0.0019*Current_Data.loc[i,'Points_For'])+(-0.0020*Current_Data.loc[i,'Points_Against'])+(-0.0117*Current_Data.loc[i,'Avg_Margin_of_Victory']) + (.0002*Current_Data.loc[i,'Turnover_Differential']) + (-.6961*Current_Data.loc[i,'Opponent_Win_Percentage'])+ (-0.0043*Current_Data.loc[i,'Time_Of_Possession'])
    return win_percentage
	
Results = pd.DataFrame(index=['ARI', 'ATL','BAL','BUF','CAR','CHI','CIN','CLE','DAL','DEN','DET','GB','HOU','IND','JAX','KC','LAC','LAR','MIA','MIN','NE','NO','NYG','NYJ','OAK','PHI','PIT','SEA','SF','TB','TEN','WAS'], columns=['Odds_of_Winning_the_Super_Bowl'])



for i in Prediction_Data.index:
        Results.loc[Current_Data.loc[i,'Teams'],'Odds_of_Winning_the_Super_Bowl'] = Super_Bowl_Odds(Prediction_Data)
		
		Results.to_excel('Prediction_Data_Results.xlsx')
