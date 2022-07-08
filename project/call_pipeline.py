
#%%

import pandas as pd
from pipeline import *
from RDS_credentials import *

#%%

def load_and_clean():
    '''
    Load datasets and call methods in pipeline.py to clean the dataframes
    '''
    # load dataset
    df = pd.read_csv('to_predict_unclean.csv')
    df = drop_unnamed(df)
    try:
        df = format_results(df)
        df = format_capacity(df)
        #df = team_names(df)
    except:
        pass
    return df

#%%

def feature_eng(df):
    '''
    This method calls the feature engineering methods in pipeline.py and returns a 
    dataframe ready for making predictions.
    '''
    #df = none_na(df)
    #df = drop_columns(df)
    df = sort_date(df) 
    #df = format_date(df)
    df = split_result(df)
    df = total_goals(df)
    df['Outcome'] = df.apply(outcome, axis=1)
    # df = call_home_streaks(df)
    # df = call_away_streaks(df)
    df['Home_Win'] = df.apply(home_win, axis=1)
    df = attack_defence(df)
    df = call_rolling(df)
    df = replace_inf(df)
    df = drop_na_features(df)
    df = drop_features(df)
    return df


#%%

df = load_and_clean()
#upload to RDS
df.to_sql("to_predict_clean", con=engine, schema='public', index=False, if_exists='append')


#%%

#pull cleaned dataset from RDS
clean_df = pd.read_sql_query('''SELECT * FROM to_predict_clean''', engine)
#perform feature engineering
final_df = feature_eng(clean_df)
#save to csv
final_df.to_csv('cleaned_results_predict.csv')


#%%

#save to RDS
final_df.to_sql("to_predict_eng", con=engine, schema='public', index=False, if_exists='append')


