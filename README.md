
# Football Match Outcome Prediction

## Contents

#### Introduction
#### Milestones 1-5
#### Conclusion

## Introduction

Sports betting corporations try to give accurate predictions of the outcome of upcoming matches. It's important to understand the data and the prediction model to get accurate predictions. This is an implementation of a data science pipeline that predicts the outcome of a football match.

The success is judged using the following two objectives, one quantitative and one qualitative:

    - Achieve a test accuracy of greater than 50%, with a stretch target of 60%.
    - Output probabilities that appear sensible/realistic, that are comparable to odds offered on popular betting websites.

## Milestone 1: EDA and Cleaning

Exploratory data analysis is a critical precursor to applying a model. It is important to identify the features in the dataset that will be necessary for predicting the outcome of a football match. 

First, visualise the data to determine the significance towards predicting match results and outcome. 

#### Total goals per Season per League:

![alt text](/figures/goalleague.png)

#### Average Home/Away Goals per Season in the Premier League:

![alt text](/figures/hagoals.png)

#### League data:

|    |     League       | No._Teams | Sample_Size | No._Home_Goals | No._Away_Goals | No._Goals | Goals/Game | No._Draws | Draws/Game |
|----|------------------|-----------|-------------|----------------|----------------|-----------|------------|-----------|------------|
| 0  |   primeira_liga  |     43	|    8542	  |     12440.0	   |     8543.0     |  20983.0	|  2.456450	 |    2224	 |  0.260361  |
| 1  |     serie_b	    |     78	|    10038	  |     13958.0	   |     10089.0	|  24047.0	|  2.395597	 |    3264	 |  0.325164  |
| 2	 |    eredivisie	|     32	|    9135	  |     16143.0	   |     11563.0	|  27706.0	|  3.032950	 |    2236	 |  0.244773  |
| 3  |     ligue_2      |     70	|    8766	  |     11837.0	   |     8308.0	    |  20145.0	|  2.298084	 |    2759	 |  0.314739  | 
| 4	 |   segunda_liga   |     39    |	 111	  |     146.0	   |     108.0	    |  254.0	|  2.288288	 |    25	 |  0.225225  |
| 5	 | segunda_division |	  60    |	 3965	  |     5347.0	   |     3828.0	    |  9175.0	|  2.313997	 |    1189	 |  0.299874  | 
| 6	 |     serie_a	    |     50	|    9134	  |     13893.0	   |     10106.0	|  23999.0	|  2.627436	 |    2572	 |  0.281585  |
| 7	 | premier_league	|     52	|    12155	  |     18579.0	   |     13715.0	|  32294.0	|  2.656849	 |    3166	 |  0.260469  |
| 8	 | primera_division |	  46	|    8839	  |     13756.0	   |     9872.0	    |  23628.0	|  2.673153	 |    2253	 |  0.254893  |
| 9	 |    bundesliga    |	  46	|    9570	  |     15929.0	   |     11650.0	|  27579.0	|  2.881818	 |    2521	 |  0.263427  |
| 10 |      2_liga      |	  83	|    9499	  |     15022.0	   |     10693.0	|  25715.0	|  2.707127	 |    2729	 |  0.287293  |
| 11 |   championship   |	  50	|    2568	  |     3773.0	   |     2848.0	    |  6621.0	|  2.578271	 |    677	 |  0.263629  |
| 12 |  eerste_divisie	|     33	|    128	  |     208.0	   |     174.0	    |  382.0	|  2.984375	 |    41	 |  0.320312  |
| 13 |     ligue_1	    |     45	|    11023	  |     15719.0	   |     10562.0	|  26281.0	|  2.384197	 |    3190	 |  0.289395  |

We expect these features to be important towards predicting outcome:

- ELO rating
- Result
- Season/Date
- Yellow/Red Cards
- Capacity

Null Hypothesis: H0: The accuracy of a football match predictor is no better than random choice.

## Milestone 2: Feature Engineering

After cleaning the dataset, we will engineer new features from the current features available in our dataset. 

Each teams Home and Away Attack and Defensive Strengths are calculated from the number of goals. Attack Strength is the team's average number of goals, divided by the league's Average number of goals. This is calculated as a rolling average as the specific team and league's games increase. 

Team average goals from a set number of previous team matches. This is also calculated as a rolling average (the window being the set number of previous matches to average).
```
def average_last_3(df, feature, ha):
    
    df[feature+'_avg_3'] = df.groupby(ha)[feature].transform(lambda x: x.rolling(3, 1).mean())
    df[feature+'_avg_3'] = df.groupby(ha)[feature+'_avg_3'].shift(fill_value=0)
    return df
```
Team sum of goals from a set number of previous team matches. 
```
def sum_last_10(df, ha):
    
    df[ha+'_Outcome_sum_10'] = df.groupby(ha)['Outcome'].transform(lambda x: x.rolling(10, 1).sum())
    df[ha+'_Outcome_sum_10'] = df.groupby(ha)[ha+'_Outcome_sum_10'].shift(fill_value=0)
    return df
```
## Milestone 3: Upload the data to the database

The steps so far are compiled into a `pipeline.py`, where the cleaned data with new features is uploaded or upserted into an AWS RDS database. 

## Milestone 4: Model Training

First, perform an baseline model selection on the current dataset, so we have a model to perform feature selection with. 

|    Models    | Logistic Regression | Linear Regression | Binary Logistic Reg |
|--------------|---------------------|-------------------|---------------------|
|   Accuracy   |       49.68%        |       33.88%      |        60.84%       |
| Random Guess |       33.53%        |       33.02%      |        50.16%       |

We will use Logistic Regression and Binary Logistic Regression to perform feature selection

Feature heatmaps are used to find correlated features and select between those that are equally useful in affecting the outcome. 

![alt text](/figures/heatmap2.png)

Lasso Regression/L1 Regularization for Binary Logistic Regression is used to to assess feature importance to model.

|    | Feature Name             |   Alpha = 1.000000 |   Alpha = 0.100000 |   Alpha = 0.001000 |
|---:|:-------------------------|-------------------:|-------------------:|-------------------:|
|  0 | Season                   |           -0.04118 |           -0.04025 |            0       |
|  1 | Capacity                 |            0.07173 |            0.07178 |            0.05776 |
|  2 | Elo_home                 |            0.46294 |            0.44237 |            0.25016 |
|  3 | Elo_away                 |           -0.48203 |           -0.46081 |           -0.25289 |
|  4 | Home_Streak              |            0.03818 |            0.03705 |            0       |
|  5 | Away_Streak              |           -0.00955 |           -0.00906 |            0       |
|  6 | Home_Attack              |            0.07124 |            0.06961 |            0.06686 |
|  7 | Home_Defence             |            0.01913 |            0.01842 |            0       |
|  8 | Away_Attack              |           -0.09874 |           -0.09654 |           -0.07756 |
|  9 | Away_Defence             |            0.00011 |            0       |            0       |
| 10 | Elo_home_avg_3           |           -0.06735 |           -0.04534 |            0       |
| 11 | Elo_home_avg_10          |            0.00142 |            0       |            0       |
| 12 | Home_Goals_avg_3         |            0.00862 |            0.00661 |            0       |
| 13 | Home_Goals_avg_10        |            0.04749 |            0.04885 |            0.03931 |
| 14 | Elo_away_avg_3           |            0.08125 |            0.04682 |            0       |
| 15 | Elo_away_avg_10          |           -0.01373 |            0       |            0       |
| 16 | Away_Goals_avg_3         |           -0.00737 |           -0.00503 |            0       |
| 17 | Away_Goals_avg_10        |            0.00612 |            0.0027  |            0       |
| 18 | Home_Team_Outcome_sum_3  |           -0.01565 |           -0.01211 |            0       |
| 19 | Home_Team_Outcome_sum_10 |            0.09651 |            0.09523 |            0.09635 |
| 20 | Away_Team_Outcome_sum_3  |            0.00692 |            0.00875 |            0       |
| 21 | Away_Team_Outcome_sum_10 |            0.14022 |            0.13886 |            0.13576 |

From these results we can determine which features are least important and remove them. If there are important features that are highly correlated, this means it is not necessary to retain all of them and some can be dropped. For example, Home Attack and Home_Goals_avg_10 are strongly related, therefore one will be removed.

We are left with the most important features:

- Elo_home
- Elo_away
- Home_Attack
- Away_Attack
- Home_Team_Outcome_last_10
- Away_Team_Outcome_last_10
- Capacity

### Model Evaluation

Now we can train our models. Compare different classification models, perform hyperparameter tuning and determine which model produces the best performace. All models were optimised using a grid search with a k-fold cross-validation accuracy metric. The top 3 performing algorithms explored in this report are:

- Logistic Regression
- Adaboost Classifier
- Gradient Boost Classifier


|    Models    | Logistic Regression | AdaBoost Classifer | GradientBoost Classifer |
|--------------|---------------------|--------------------|-------------------------|
|   Accuracy   |       49.74%        |       49.45%       |        49.46%           |


Models were assessed for accuracy using Confusion Matrices:

<span style="color:white;"![alt text](/figures/cm_norm_lgr.png)</span>

![alt text](/figures/cm_norm_gbc.png)
![alt text](/figures/cm_norm_abc.png)

Confusion matrices show that all three models perform extremely poorly when predicting a draw. The models predict more incorrect than correct and only predict a small amount of draws from over 100k fixtures, despite the fact that draws take up over 25% of match results across all leagues (loss=-1, draw=0, win=1):

![alt text](/figures/outcomes.png)

Pairwise relationship plots between the features show that there is a stronger distinction between win or loss outcomes (and comparatively it is more difficult to make a distinction for draws), and there is a much higher frequency of wins (approx 50%) relative to draws. Therefore, this may explain why the models are more likely to predict win as an outcome.

#### [PairGrid png]

Since the models are doing a very poor job of predicting draws (near negligible) and not achieving over 50% accuracy, we will instead use a Binary Classification model and predict on home wins only. After optimisation, the top 3 performing models achieve over 60% accuracy. 


|    Models    | Logistic Regression | AdaBoost Classifer | GradientBoost Classifer |
|--------------|---------------------|--------------------|-------------------------|
|   Accuracy   |       60.93%        |       60.98%       |        60.87%           |


Confusion matrices can allow you to assess the sensitivity and specificity of the models. Interestingly, the confusion matrices (below) for these 3 models show that the 'sensitivity' of predicted wins from true wins is significantly less accurate. However, this is due to the multiclass models being heavily skewed towards predicting wins. In the case of multiclass verses binary classification models, overall accuracy may not be enough to distinguish which is the better model, and instead depends on whether there are specific classes that need to be labeled correctly more than others.  

![alt text](/figures/binary_cm_norm_lgr.png)
![alt text](/figures/binary_cm_norm_gbc.png)
![alt text](/figures/binary_cm_norm_abc.png)

As the accuracy scores for the 3 models are very similar, to determine which model is the best option, we can assess the reliablility and use significance tests to find any differences in the performance. Prediction probability can be used to show the reliablity of models, where a good model should output probabilities that are consistent with the outcome e.g. 60% probability should be correct 60% of the time. Below shows histograms of correct and incorrect predictions given a predictions probability. 

![alt text](/figures/pred_prob_lgr.png)
![alt text](/figures/pred_prob_gbc.png)
![alt text](/figures/pred_prob_abc.png)


Significance tests are used to compare models. Based on the parameters of 3 different types of statistical tests: Wilcoxon rank signed, McNemar's and combined 5x2CV F-test. We can conclude there is no significant difference in the performance of the models. 

(lgr = Logistic Regression, gbc = Gradient Boost, abc = AdaBoost)

|    Models    | Wilcoxon | McNemars | Combined 5X2cv f-test |
|--------------|----------|----------|-----------------------|
|  lgr vs gbc  |  0.1875  |  0.766   |         0.048         |
|  lgr vs abc  |  0.625   |  0.250   |         0.109         |
|  gbc vs abc  |  0.125   |  0.226   |         0.310         |


## Milestone 5: Inference

Use the `pipeline.py` to clean and create features for a new dataset from Results and To_Predict in the raw data folder, To_Predict contains information for matches that have yet to take place (no results column). 

Use the chosen model to predict the outcome of the matches from the To_Predict dataset.


## Conclusion

- To achieve over 50% accuracy, we performed Binary Classification using `Home_Win` as the prediction label.

- Modern football match prediction models utilise more in-depth match statistics such as play-by-play data including information on each shot or pass made in a match. Which, for example, has lead to the development of 'expected goals' and ELO scores for each team per match. Therefore, we expect that increasing the statistical depth on which predictions are made will improve the accuracy of the models.

- Having better match statistics may allow us to build multi-class classification models with over 50% accuracy.

https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/1718-ug-projects/Corentin-Herbinet-Using-Machine-Learning-techniques-to-predict-the-outcome-of-profressional-football-matches.pdf