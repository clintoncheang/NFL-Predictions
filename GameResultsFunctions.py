import pandas as pd
import numpy as np

# Importing the Pro Football Reference Individual Game Results spreadsheet
gameResults = pd.read_csv('NFLGameResults.csv')

# The below function calculates a number of stats for each team by season
# The code is clunky and slow due to using too many iterations through the dataset
# Could be vastly improved for performance by combining the calculations when possible
def getRegSeasonRecord(Team, Year):
    homeW = 0
    homeL = 0
    homeT = 0
    
    # Calculating home/away wins/losses/ties
    for i in gameResults.index:
        if (gameResults['Home Team'][i] == Team) & (gameResults['Season'][i] == Year) & (gameResults['Week'][i] <= 17):
            if (gameResults.loc[i,'Winner/tie'] == Team) & (gameResults['PtsW'][i] != gameResults['PtsL'][i]):
                homeW += 1
            elif (gameResults.loc[i,'Loser/tie'] == Team) & (gameResults['PtsW'][i] != gameResults['PtsL'][i]):
                homeL += 1
            else:
                homeT += 1
    
    awayW = 0
    awayL = 0
    awayT = 0
    for i in gameResults.index:
        if (gameResults.loc[i,'Away Team'] == Team) & (gameResults['Season'][i] == Year) & (gameResults['Week'][i] <= 17):
            if (gameResults.loc[i,'Winner/tie'] == Team) & (gameResults['PtsW'][i] != gameResults['PtsL'][i]): #& (gameResults['Loser/tie'][i] != Team) 
                awayW += 1
            elif (gameResults.loc[i,'Loser/tie'] == Team) & (gameResults['PtsW'][i] != gameResults['PtsL'][i]): #& (gameResults['Winner/tie'][i] != Team) 
                awayL += 1
            else:
                awayT += 1
    
    # Calculating margin of victory and defeat
    margW = 0
    margL = 0
    for i in gameResults.index:
        if (gameResults['Season'][i] == Year) & (gameResults['Week'][i] <= 17):
            if (gameResults.loc[i,'Winner/tie'] == Team):
                margW = margW + gameResults['PtsW'][i] - gameResults['PtsL'][i]
            elif (gameResults.loc[i,'Loser/tie'] == Team):
                margL = margL + gameResults['PtsW'][i] - gameResults['PtsL'][i]
    
    ptsScored = 0
    ptsAgainst = 0
    for i in gameResults.index:
        if (gameResults.loc[i,'Home Team'] == Team) & (gameResults['Season'][i] == Year) & (gameResults['Week'][i] <= 17):
            if (gameResults.loc[i,'Winner/tie'] == Team):
                ptsScored = ptsScored + gameResults['PtsW'][i]
                ptsAgainst = ptsAgainst + gameResults['PtsL'][i]
            elif (gameResults.loc[i,'Loser/tie'] == Team):
                ptsScored = ptsScored + gameResults['PtsL'][i]
                ptsAgainst = ptsAgainst + gameResults['PtsW'][i]
            else:
                ptsScored = ptsScored + gameResults['PtsW'][i]
                ptsAgainst = ptsAgainst + gameResults['PtsL'][i]
                
        elif (gameResults.loc[i,'Away Team'] == Team) & (gameResults['Season'][i] == Year) & (gameResults['Week'][i] <= 17):
            if (gameResults.loc[i,'Winner/tie'] == Team):
                ptsScored = ptsScored + gameResults['PtsW'][i]
                ptsAgainst = ptsAgainst + gameResults['PtsL'][i]
            elif (gameResults.loc[i,'Loser/tie'] == Team):
                ptsScored = ptsScored + gameResults['PtsL'][i]
                ptsAgainst = ptsAgainst + gameResults['PtsW'][i]
            else:
                ptsScored = ptsScored + gameResults['PtsW'][i]
                ptsAgainst = ptsAgainst + gameResults['PtsL'][i]
    
    wins = float(homeW + awayW)
    losses = float(homeL + awayL)
    ties = float(homeT + awayT)
    games = float(wins + losses + ties)
    avgMargW = float(0)
    avgMargL = float(0)
    
    if wins > 0:
        avgMargW = float(margW/wins)
    
    if losses > 0:
        avgMargL = float(margL/losses)
    
    # Enters all above calculated data into a series to be imported into a later dataframe
    winLoss = float(wins/games)
    record = pd.Series(data=[Team,Year,homeW,homeL,homeT,awayW,awayL,awayT,wins,losses,ties,winLoss,ptsScored,ptsAgainst,avgMargW,avgMargL])
    return record
   
# This function calaculates the league's schedule by season
# Returns a complete 32 x 16 dataframe for all teams
# Does not include bye week information
def getSchedule(Year):
    teams = pd.Series()
    if Year == 2017:
        teams = pd.Series(data=["Arizona Cardinals", "Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
        "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions",
        "Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
        "Los Angeles Chargers","Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots",
        "New Orleans Saints","New York Giants","New York Jets","Oakland Raiders","Philadelphia Eagles","Pittsburgh Steelers",
        "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Redskins"])
    
    elif Year == 2016:
        teams = pd.Series(data=["Arizona Cardinals", "Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
        "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions",
        "Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
        "Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots","New Orleans Saints",
        "New York Giants","New York Jets","Oakland Raiders","Philadelphia Eagles","Pittsburgh Steelers","San Diego Chargers",
        "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Redskins"])
    
    else:
        teams = pd.Series(data=["Arizona Cardinals", "Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
        "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions",
        "Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
        "Miami Dolphins","Minnesota Vikings","New England Patriots","New Orleans Saints", "New York Giants",
        "New York Jets","Oakland Raiders","Philadelphia Eagles","Pittsburgh Steelers","San Diego Chargers","San Francisco 49ers",
        "Seattle Seahawks","St. Louis Rams","Tampa Bay Buccaneers","Tennessee Titans","Washington Redskins"])
    
    yearSchedule = pd.DataFrame()
    opponent = ""
    yearSchedule = pd.DataFrame(teams).reset_index(drop = True)
    for i in yearSchedule.index:
        Team = yearSchedule.loc[i,0]
        k = 1
        for j in gameResults.index:
            if (gameResults.loc[j,'Home Team'] == Team) & (gameResults.loc[j,'Season'] == Year) & (gameResults.loc[j,'Week'] <= 17):
                opponent = gameResults.loc[j,'Away Team']
                yearSchedule.at[i, k] = opponent
                k += 1
            elif (gameResults.loc[j,'Away Team'] == Team) & (gameResults.loc[j,'Season'] == Year) & (gameResults.loc[j,'Week'] <= 17):
                opponent = gameResults.loc[j,'Home Team']
                yearSchedule.at[i, k] = opponent
                k += 1
    return yearSchedule
    

# This function calculates opponent win percentage by adding up wins/losses/ties from the GetSeasonRecordFunction
# Compiles all calculated stats into a single dataframe per season
# Then exports each datafram to it's own CSV file
def writeToExcel():
    years = pd.Series(data=[2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017])
    teams1 = pd.Series(data=["Arizona Cardinals", "Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
    "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions",
    "Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
    "Miami Dolphins","Minnesota Vikings","New England Patriots","New Orleans Saints", "New York Giants",
    "New York Jets","Oakland Raiders","Philadelphia Eagles","Pittsburgh Steelers","San Diego Chargers","San Francisco 49ers",
    "Seattle Seahawks","St. Louis Rams","Tampa Bay Buccaneers","Tennessee Titans","Washington Redskins"])
    teams2 = pd.Series(data=["Arizona Cardinals", "Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
    "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions",
    "Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
    "Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots","New Orleans Saints",
    "New York Giants","New York Jets","Oakland Raiders","Philadelphia Eagles","Pittsburgh Steelers","San Diego Chargers",
    "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Redskins"])
    teams3 = pd.Series(data=["Arizona Cardinals", "Atlanta Falcons","Baltimore Ravens","Buffalo Bills","Carolina Panthers",
    "Chicago Bears","Cincinnati Bengals","Cleveland Browns","Dallas Cowboys","Denver Broncos","Detroit Lions",
    "Green Bay Packers","Houston Texans","Indianapolis Colts","Jacksonville Jaguars","Kansas City Chiefs",
    "Los Angeles Chargers","Los Angeles Rams","Miami Dolphins","Minnesota Vikings","New England Patriots",
    "New Orleans Saints","New York Giants","New York Jets","Oakland Raiders","Philadelphia Eagles","Pittsburgh Steelers",
    "San Francisco 49ers","Seattle Seahawks","Tampa Bay Buccaneers","Tennessee Titans","Washington Redskins"])
    
    recordsDF = pd.DataFrame()
    schedule = pd.DataFrame()
    
    for i in years.index:
        recordsDF = pd.DataFrame()
        schedule = pd.DataFrame()
    
        schedule = getSchedule(years[i])
        if years[i] == 2016:
            for j in teams2.index:
                record = getRegSeasonRecord(teams2[j], years[i])
                record = record.to_frame().transpose()
                recordsDF = recordsDF.append(record)
                
        elif years[i] == 2017:
            for j in teams3.index:
                record = getRegSeasonRecord(teams3[j], years[i])
                record = record.to_frame().transpose()
                recordsDF = recordsDF.append(record)
        else:
            for j in teams1.index:
                record = getRegSeasonRecord(teams1[j], years[i])
                record = record.to_frame().transpose()
                recordsDF = recordsDF.append(record)
        
        recordsDF['oppWinPer'] = ''
        
        for k in schedule.index:
            oppWin = 0
            oppLoss = 0
            oppTies = 0
            for c in range(1,17):
                opponent = schedule.loc[k, c]
                oppWin = float(oppWin + recordsDF[recordsDF[0]==opponent][8])
                oppLoss = float(oppLoss + recordsDF[recordsDF[0]==opponent][9])
                oppTies = float(oppTies + recordsDF[recordsDF[0]==opponent][10])
            oppWinPer = oppWin / (oppWin + oppLoss + oppTies)
            recordsDF.iloc[k,16] = oppWinPer
        
        if years[i] == 2007:
            recordsDF.to_csv(path_or_buf='NFLResults07.csv')
        elif years[i] == 2008:
            recordsDF.to_csv(path_or_buf='NFLResults08.csv')
        elif years[i] == 2009:
            recordsDF.to_csv(path_or_buf='NFLResults09.csv')
        elif years[i] == 2010:
            recordsDF.to_csv(path_or_buf='NFLResults10.csv')
        elif years[i] == 2011:
            recordsDF.to_csv(path_or_buf='NFLResults11.csv')
        elif years[i] == 2012:
            recordsDF.to_csv(path_or_buf='NFLResults12.csv')
        elif years[i] == 2013:
            recordsDF.to_csv(path_or_buf='NFLResults13.csv')
        elif years[i] == 2014:
            recordsDF.to_csv(path_or_buf='NFLResults14.csv')
        elif years[i] == 2015:
            recordsDF.to_csv(path_or_buf='NFLResults15.csv')
        elif years[i] == 2016:
            recordsDF.to_csv(path_or_buf='NFLResults16.csv')
        elif years[i] == 2017:
            recordsDF.to_csv(path_or_buf='NFLResults17.csv') 