
# coding: utf-8

# In[4]:

#APPENDIX B - PYTHON CODE AND ASSOCIATED OUTPUT FOR SECTION 5.0 DATA UNDERSTANDING

import pandas as pd
import numpy as np
from __future__ import division
from pydoc import help
from scipy.stats import pearsonr
import scipy.spatial.distance as dist
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Import Test Data and Drop Category Fields
df = pd.read_excel("Master NFL File.xlsx")
df = df.drop(['Season', 'Away_Ties', 'Total_Ties', 'Home_Ties'], 1)

# Calculate Summary Statistics
np.round(df.describe(), 2).T

# Calculate Correlation Coeffiecients and Heat Map
cortab = np.round(df.corr(),2)
cortab.to_csv('corrtable.csv')
cortab
sns.heatmap(np.round(df.corr(),2),annot=False, fmt="g", cmap='viridis')
plt.show()


# Create Scatter Plot Matrix
sns.pairplot(df[["Win_Percentage", "Turnover_Differential","Sacks_Allowed","Sacks _Gained",
                "Tackles_for_Loss","Offensive_Yards_Gained","Defensive_Yards_Allowed",
                "Number_Of_Pro_Bowl_Players","Time_Of_Possession", "Home_Wins", "Home_Losses","Away_Wins", "Away_Losses", 
                "Total_Wins", "Total_Losses","Points_For", "Points_Against", "Avg_Margin_of_Victory",
                "Avg_Margin_of_Defeat","Opponent_Win_Percentage"]], diag_kind="hist")
plt.subplots_adjust(top=0.9)
#sns.plt.suptitle('Figure 1 - Correlation Matrix', fontsize=50)
sns.plt.show()

# Create Individual Scatter Plots w/ Correlation Coefficients - vs Win Percentage
colors = (0,0,0)
area = np.pi*3
plt.scatter(df['Win_Percentage'],df['Turnover_Differential'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Turnover_Differential'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Turnover_Differential'])
rval = round(r_row,3)
pval = round(p_value, 6)
plt.text(0, 30, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1a')
plt.xlabel('Win %')
plt.ylabel('Turnover Diff')
sns.jointplot("Win_Percentage", "Turnover_Differential", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Sacks_Allowed'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Sacks_Allowed'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Sacks_Allowed'])
rval = round(r_row,3)
pval = round(p_value, 6)
plt.text(0, 70, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1b')
plt.xlabel('Win %')
plt.ylabel('Sacks Allowed')
sns.jointplot("Win_Percentage", "Sacks_Allowed", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Sacks _Gained'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Sacks _Gained'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Sacks _Gained'])
rval = round(r_row,3)
pval = round(p_value, 6)
plt.text(0, 60, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1c')
plt.xlabel('Win %')
plt.ylabel('Sacks Gained')
sns.jointplot("Win_Percentage", "Sacks _Gained", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Tackles_for_Loss'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Tackles_for_Loss'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Tackles_for_Loss'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.02, 80, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1d')
plt.xlabel('Win %')
plt.ylabel('Tackles for Loss')
sns.jointplot("Win_Percentage", "Tackles_for_Loss", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Offensive_Yards_Gained'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Offensive_Yards_Gained'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Offensive_Yards_Gained'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 7500, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1e')
plt.xlabel('Win %')
plt.ylabel('Offensive Yards Gained')
sns.jointplot("Win_Percentage", "Offensive_Yards_Gained", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Defensive_Yards_Allowed'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Defensive_Yards_Allowed'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Defensive_Yards_Allowed'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 7000, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1f')
plt.xlabel('Win %')
plt.ylabel('Defensive Yards Allowed')
sns.jointplot("Win_Percentage", "Defensive_Yards_Allowed", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Number_Of_Pro_Bowl_Players'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Number_Of_Pro_Bowl_Players'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Number_Of_Pro_Bowl_Players'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 12, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1g')
plt.xlabel('Win %')
plt.ylabel('Number Of ProBowl Players')
sns.jointplot("Win_Percentage", "Number_Of_Pro_Bowl_Players", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Time_Of_Possession'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Time_Of_Possession'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Time_Of_Possession'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 33, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1h')
plt.xlabel('Win %')
plt.ylabel('Time Of Possession')
sns.jointplot("Win_Percentage", "Time_Of_Possession", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Points_For'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Points_For'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Points_For'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 600, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1i')
plt.xlabel('Win %')
plt.ylabel('Points For')
sns.jointplot("Win_Percentage", "Points_For", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Points_Against'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Points_Against'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Points_Against'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 500, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1j')
plt.xlabel('Win %')
plt.ylabel('Points Against')
sns.jointplot("Win_Percentage", "Points_Against", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Avg_Margin_of_Victory'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Avg_Margin_of_Victory'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Avg_Margin_of_Victory'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 20, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1k')
plt.xlabel('Win %')
plt.ylabel('Avg Margin of Victory')
sns.jointplot("Win_Percentage", "Avg_Margin_of_Victory", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Avg_Margin_of_Defeat'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Avg_Margin_of_Defeat'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Avg_Margin_of_Defeat'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, 20, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1l')
plt.xlabel('Win %')
plt.ylabel('Avg Margin of Defeat')
sns.jointplot("Win_Percentage", "Avg_Margin_of_Defeat", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Opponent_Win_Percentage'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Opponent_Win_Percentage'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Opponent_Win_Percentage'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1m')
plt.xlabel('Win %')
plt.ylabel('Opponent Win Percentage')
sns.jointplot("Win_Percentage", "Opponent_Win_Percentage", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Total_Wins'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Total_Wins'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Total_Wins'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1n')
plt.xlabel('Win %')
plt.ylabel('Total_Wins')
sns.jointplot("Win_Percentage", "Total_Wins", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Total_Losses'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Total_Losses'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Total_Losses'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1o')
plt.xlabel('Win %')
plt.ylabel('Total_Losses')
sns.jointplot("Win_Percentage", "Total_Losses", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Home_Wins'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Home_Wins'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Home_Wins'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1p')
plt.xlabel('Win %')
plt.ylabel('Home_Wins')
sns.jointplot("Win_Percentage", "Home_Wins", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Home_Losses'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Home_Losses'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Home_Losses'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1q')
plt.xlabel('Win %')
plt.ylabel('Home_Losses')
sns.jointplot("Win_Percentage", "Home_Losses", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Away_Wins'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Away_Wins'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Away_Wins'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1r')
plt.xlabel('Win %')
plt.ylabel('Away_Wins')
sns.jointplot("Win_Percentage", "Away_Wins", data=df, kind='kde')
sns.plt.show()

plt.scatter(df['Win_Percentage'],df['Away_Losses'],s=area, c=colors, alpha=0.5)
sns.regplot(df['Win_Percentage'],df['Away_Losses'])
r_row, p_value = pearsonr(df['Win_Percentage'],df['Away_Losses'])
rval = round(r_row,3)
pval = round(p_value, 4)
plt.text(0.0, .6, 'r= %s'%(rval) + ' (p=%s'%(pval)+')', ha='center', va='center')
plt.title('Figure 1s')
plt.xlabel('Win %')
plt.ylabel('Away_Losses')
sns.jointplot("Win_Percentage", "Away_Losses", data=df, kind='kde')
sns.plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



