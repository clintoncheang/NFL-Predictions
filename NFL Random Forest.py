from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

np.random.seed(0)

# Establishing the training data as the season 2007 - 2015
# From online research, it appears Random Forest does not need to validate 
dataset = pd.read_excel('Master NFL File.xlsx', sheetname = 0)

Training_Data= dataset[dataset.Season < 2016]

# The test season is 2016
Test_Data = dataset[dataset.Season == 2016]

football_stats = Training_Data.columns[5:14]

# Our current attempt at the Random Forest Regressor model
Football_Regressor = RandomForestRegressor(n_jobs=5, random_state=0,bootstrap=True,n_estimators=150)

yR = Training_Data['Win Percentage']

Football_Regressor.fit(Training_Data[football_stats], yR)

expectedR = Football_Regressor.predict(Test_Data[football_stats])

# R squared was 0.977, which is concerningly high
# We need to run more tests to feel confident stating the R squared as fact
Football_Regressor.score(Training_Data[football_stats], yR,)