
import pandas as pd
import numpy as np

#%%

'''
This is a pipeline to clean the football datasets and upsert into RDS database,
so they are ready to be used in the prediction models.
'''

def drop_unnamed(df):
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df

def format_results(df):
    '''
    This method find inconsistent results data and removes it from the
    '''
    possible_results = []
    for i in range(20):
        for j in range(20):
            possible_results.append(f'{i}-{j}')
    df = df.drop(df.loc[~df['Result'].isin(possible_results)].index)
    
    return df

def format_capacity(df):
    '''
    This method is to make sure Capacity column values have the same format
    '''
    df['Capacity'] = df['Capacity'].str.replace(',', '')
    df['Capacity'] = df['Capacity'].astype('float')
    return df

def team_names(df):
    '''
    This method removes inconsistent teams between home and away team columns
    '''
    diff_home = np.setdiff1d(df['Home_Team'], df['Away_Team'])
    diff_away = np.setdiff1d(df['Away_Team'], df['Home_Team'])
    df.drop(df[df['Home_Team'].isin(diff_home)].index, inplace=True)
    df.drop(df[df['Away_Team'].isin(diff_away)].index, inplace=True)
    
    return df

def drop_na(df):
    '''
    This method removes rows that contain NA values
    '''
    df = df.dropna(axis=0)
    return df


#%%

'''
This portion of the pipeline is for feature engineering.

'''

def none_na(df):
    '''
    This method replaces None values with NaN
    '''
    df = df.replace(to_replace=None, value=np.nan, inplace=True)
    return df

def drop_columns(df):
    '''
    This method drops columns that arent necessary for feature engineering onward.
    '''
    
    df = df.drop(['Home_Yellow', 'Home_Red', 'Away_Yellow', 'Away_Red', 
                    'Pitch', 'Referee', 'City', 'Country', 'Stadium'], axis=1)
    
    return df

def format_date(df):
    '''
    This method edits Date_New into usable date format and sorts the rows by date.
    '''
    df['Date'] = df['Date_New'].apply(lambda x: x[x.find(',') + 2:x.rfind(',')])
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df.drop(['Date_New'], axis=1)
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    return df

def sort_date(df):
    '''
    This method sorts columns into descending date order using 'Season' and 'Round'
    columns.
    '''
    # sort descending date
    df = df.sort_values(['Season', 'Round'], ascending=True).reset_index(drop=True)
    return df

def split_result(df):
    '''
    This method splits Result column into Home Goals and Away Goals.
    It also creates a Total_Goals column from Home and Away Goals.
    '''
    df[['Home_Goals', 'Away_Goals']] = df['Result'].str.split('-', expand=True)
    
    df['Away_Goals'] = df['Away_Goals'].astype('float', errors='ignore')
    df['Home_Goals'] = df['Home_Goals'].astype('float', errors='ignore')
    
    return df

def total_goals(df):
    '''
    This method creates Total_goals in a game from Home_Goals and Away_Goals 
    columns
    '''
    #create total goals column
    df['Total_goals'] = df['Home_Goals'] + df['Away_Goals']
    return df

def outcome(df):
    '''
    This method creates an Outcome column from the Home_Goals and Away_Goals columns
    '''
    if df['Home_Goals'] != None:
        
        if df['Home_Goals'] > df['Away_Goals']:
            return 1
            
        elif df['Home_Goals'] == df['Away_Goals']:
            return 0
            
        elif df['Home_Goals'] < df['Away_Goals']:
            return -1

def home_streaks(df):
    '''
    This method creates a Home_Streak column
    '''
    s = df['Outcome'].groupby((df['Outcome'] != df['Outcome'].shift()).cumsum()).cumsum()
    return df.assign(Home_Streak=s.where(s>0, 0.0).abs())

def call_home_streaks(df):
    '''
    This method calls the home_streaks method
    '''
    df = df.groupby('Home_Team').apply(home_streaks)
    #shift all colunms forward to next game
    df['Home_Streak'] = df.groupby('Home_Team')['Home_Streak'].shift(fill_value=0)
    return df

def away_streaks(df):
    '''
    This method creates an Away_Streak column
    '''
    s = df['Outcome'].groupby((df['Outcome'] != df['Outcome'].shift()).cumsum()).cumsum()
    return df.assign(Away_Streak=s.where(s<0, 0.0).abs())

def call_away_streaks(df):
    '''
    This method calls the away streaks method
    '''
    df = df.groupby('Away_Team').apply(away_streaks)
    #shift all colunms forward to next game
    df['Away_Streak'] = df.groupby('Away_Team')['Away_Streak'].shift(fill_value=0)
    return df

def home_win(df):
    '''
    This method creates a Home_Win column from Outcome.
    (This is for Binary Classification later)
    '''
    if df['Outcome'] == 1.0:
        return 1
    elif (df['Outcome'] == -1.0 or df['Outcome'] == 0.0):
        return 0
    else:
        return np.nan

def attack_defence(df):
    '''
    This method creates an attack strength and defence strength for home and 
    away teams.
    '''
    # cumulative goals per team
    df['Cum_Home_Goals'] = df.groupby(['Home_Team'])['Home_Goals'].cumsum()
    df['Cum_Away_Goals'] = df.groupby(['Away_Team'])['Away_Goals'].cumsum()
    # cumulative games played per team
    df['Home_Games_so_far'] = df.groupby(['Home_Team']).cumcount() + 1
    df['Away_Games_so_far'] = df.groupby(['Away_Team']).cumcount() + 1
    # cumulative average home \ away goals scored per team
    df['Cum_Avg_Home_Goals'] = df['Cum_Home_Goals'] / df['Home_Games_so_far']
    df['Cum_Avg_Away_Goals'] = df['Cum_Away_Goals'] / df['Away_Games_so_far']
    
    # Cumulative Home/Away goals/games so far ALL Leagues
    df['League_Home_Goals'] = df.groupby('League')['Home_Goals'].cumsum()
    df['League_Away_Goals'] = df.groupby('League')['Away_Goals'].cumsum()
    df['League_Games'] = df.groupby('League')['Home_Goals'].cumcount() + 1
    
    df['League_Avg_Home_Goal'] = df['League_Home_Goals'] / df['League_Games']
    df['League_Avg_Away_Goal'] = df['League_Away_Goals'] / df['League_Games']
    
    #Attack/Defence Strength
    df['Home_Attack'] = (df['Cum_Home_Goals'] / df['Home_Games_so_far']) / df['League_Avg_Home_Goal']
    df['Home_Defence'] = (df['Cum_Away_Goals'] / df['Home_Games_so_far']) / df['League_Avg_Away_Goal']
    df['Away_Attack'] = (df['Cum_Away_Goals'] / df['Away_Games_so_far']) / df['League_Avg_Away_Goal']
    df['Away_Defence'] = (df['Cum_Home_Goals'] / df['Away_Games_so_far']) / df['League_Avg_Home_Goal']
    
    #shift all colunms forward to next game
    df['Home_Attack'] = df.groupby('Home_Team')['Home_Attack'].shift(fill_value=0)
    df['Home_Defence'] = df.groupby('Home_Team')['Home_Defence'].shift(fill_value=0)
    df['Away_Attack'] = df.groupby('Away_Team')['Away_Attack'].shift(fill_value=0)
    df['Away_Defence'] = df.groupby('Away_Team')['Away_Defence'].shift(fill_value=0)
    
    #drop unneeded columns
    df = df.drop(['Total_goals', 'Cum_Home_Goals', 'Cum_Away_Goals', 'Home_Games_so_far', 'Away_Games_so_far',
                'Cum_Avg_Home_Goals', 'Cum_Avg_Away_Goals', 'League_Avg_Home_Goal', 'League_Avg_Away_Goal',
                'League_Home_Goals', 'League_Away_Goals', 'League_Games'], axis=1)
    
    return df

def average_last_3(df, feature, ha):
    '''
    This method calculates the rolling average of the last three features
    '''
    df[feature+'_avg_3'] = df.groupby(ha)[feature].transform(lambda x: x.rolling(3, 1).mean())
    df[feature+'_avg_3'] = df.groupby(ha)[feature+'_avg_3'].shift(fill_value=0)
    return df

def average_last_10(df, feature, ha):
    '''
    This method calculates the rolling average of the last ten features
    '''
    df[feature+'_avg_10'] = df.groupby(ha)[feature].transform(lambda x: x.rolling(10, 1).mean())
    df[feature+'_avg_10'] = df.groupby(ha)[feature+'_avg_10'].shift(fill_value=0)
    return df

def sum_last_3(df, ha):
    '''
    This method calculates the rolling sum of the last three features
    '''
    df[ha+'_Outcome_sum_3'] = df.groupby(ha)['Outcome'].transform(lambda x: x.rolling(3, 1).sum())
    df[ha+'_Outcome_sum_3'] = df.groupby(ha)[ha+'_Outcome_sum_3'].shift(fill_value=0)
    return df

def sum_last_10(df, ha):
    '''
    This method calculates the rolling sum of the last ten features
    '''
    df[ha+'_Outcome_sum_10'] = df.groupby(ha)['Outcome'].transform(lambda x: x.rolling(10, 1).sum())
    df[ha+'_Outcome_sum_10'] = df.groupby(ha)[ha+'_Outcome_sum_10'].shift(fill_value=0)
    return df

def call_rolling(df):
    '''
    This method calls the previous methods average_last for Home_Goals, 
    Away_Goals, Elo_home, Elo_away. It also calls sum_last functions for
    Outcome for Home and Away teams.
    '''
    average_last_3(df, 'Home_Goals', 'Home_Team')
    average_last_3(df, 'Elo_home', 'Home_Team')
    average_last_10(df, 'Home_Goals', 'Home_Team')
    average_last_10(df, 'Elo_home', 'Home_Team')
        
    average_last_3(df, 'Away_Goals', 'Away_Team')
    average_last_3(df, 'Elo_away', 'Away_Team')
    average_last_10(df, 'Away_Goals', 'Away_Team')
    average_last_10(df, 'Elo_away', 'Away_Team')
        
    sum_last_3(df, 'Home_Team')
    sum_last_10(df, 'Home_Team')
    sum_last_3(df, 'Away_Team')
    sum_last_10(df, 'Away_Team')
    
    return df

def replace_inf(df):
    '''
    This method converts infinite values into NA values.
    '''
    # replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def drop_na_features(df):
    '''
    This method drops the columns that contain NA values that are not in the original
    to predict blank set
    '''
    
    df = df.dropna(subset=['Home_Attack', 'Home_Defence', 'Away_Attack', 'Away_Defence'], axis=0)
    return df

def drop_features(df):
    '''
    This method drops features that were not important
    for predicting outcome during feature selection.
    '''
    df = df.drop(['Elo_home_avg_3',
                'Elo_home_avg_10',
                'Elo_away_avg_3',
                'Elo_away_avg_10',
                'Home_Goals_avg_3',
                'Away_Goals_avg_3',
                'Home_Defence',
                'Away_Defence',
                'Home_Team_Outcome_sum_3',
                'Away_Team_Outcome_sum_3',
                'Home_Goals_avg_10',
                'Away_Goals_avg_10'], axis=1)
    return df


# %%

def drop_final(df):
    '''
    This method drops all columns except the features found to be most important
    for predicting outcome during feature selection.
    '''
    df = df.drop(['Home_Team', 
                'Away_Team', 
                'Result', 
                'League', 
                'Home_Goals',
                'Away_Goals', 
                'Date_New', 
                'Season',
                'Round',
                'Link'], axis=1)
    return df