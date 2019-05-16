import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import statsmodels.api as stats
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# Read in data file
df1 = pd.read_excel("Master NFL File.xlsx")

# Select training data
Temp_Season = df1[df1.Season < 2014]    
df = Temp_Season[['Win_Percentage', 'Offensive_Yards_Gained']]

# Normaliztion
df = df.apply(lambda x: (x - np.mean(x)) / (np.std(x)))

# Inspect data frame
df.head()

# Calculate error
def computewinpercentage(X, y, theta):  
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

# Insert ones column
df.insert(0, 'Ones', 1)

# Select Dependent and Independent Matrixes
cols = df.shape[1]
X = df.iloc[:,[0,2]]
y = df.iloc[:,[1]]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# Gradient Descent
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

# Select alpha and iterations
alpha = 0.01
iterations = 1000

g, win = gradientDescent(X, y, theta, alpha, iterations)

computewinpercentage(X, y, g)

# Plot best fit & iterations
x = np.linspace(df['Win_Percentage'].min(), df['Win_Percentage'].max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(df['Win_Percentage'], df.Offensive_Yards_Gained, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Win Percentage')
ax.set_ylabel('Offensive Yards')
ax.set_title('Win Percentage vs. Offensive Yards')

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iterations), win, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Wins')
ax.set_title('Error vs. Training Epoch')


#