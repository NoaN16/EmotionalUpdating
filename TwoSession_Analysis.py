# -*- coding: utf-8 -*-
"""
Article Code: "Dynamic emotional updating as a computational marker of well-being"
@Code Author: Noa Nagar


Code 4- Two Session Analysis
Data analyses for the test-retest study (two sessions experiment) (ratings and model).
The two-session dataset was taken from the open-access Jangraw et al. (2023) OSF repository. Here we used only participants who completed the closed-loop task in two consecutive sessions.

First, download the files 'SessionOne_ParticipantsRatings' (for the first session), 'SessionTwo_ParticipantsRatings' (for the second session) and 'ModelResults' from the 'twoSessionsData' folder.

Then, execute this code.
"""

#%% Import Libraries

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import pingouin as pg

#%% Import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "twoSessionsData")

# Session one
FirstSession_DIR = os.path.join(DATA_DIR, "SessionOne_ParticipantsRatings.xlsx")
first_df = pd.read_excel(FirstSession_DIR)

# Session two
SecondSession_DIR = os.path.join(DATA_DIR, "SessionTwo_ParticipantsRatings.xlsx")
second_df = pd.read_excel(SecondSession_DIR)

# Model Results
ModelResults_DIR = os.path.join(DATA_DIR, "ModelResults.xlsx")
modelresults_df = pd.read_excel(ModelResults_DIR)

# Data preparation
first_df = first_df.rename(columns = {'MTurkID': 'participant_id'})
second_df = second_df.rename(columns = {'MTurkID': 'participant_id'})


# Pivot the dataframe so that each participant has a row with betaR values from session 1 and 2
betaR_df = modelresults_df.pivot(index='participant_id', columns='round', values='betaR')
betaR_df.columns = ['betaR_1', 'betaR_2']
betaR_df = betaR_df.dropna().reset_index()

# Pivot the dataframe so that each participant has a row with betaP values from session 1 and 2
betaP_df = modelresults_df.pivot(index='participant_id', columns='round', values='betaP')
betaP_df.columns = ['betaP_1', 'betaP_2']
betaP_df = betaP_df.dropna().reset_index()

#%% Ratings analysis

# Correlation in mean mood across sessions

first_MeanMood = first_df.groupby('participant_id')['happySlider.response'].mean().reset_index(name='mean_ratings')
second_MeanMood = second_df.groupby('participant_id')['happySlider.response'].mean().reset_index(name= 'mean_ratings')
MeanMood = pd.merge(first_MeanMood, second_MeanMood, on='participant_id', suffixes=('_1', '_2'))

# Pearson correlation
corr_meanmood, p_meanmood = stats.pearsonr(MeanMood['mean_ratings_1'], MeanMood['mean_ratings_2'])
print(f"\nPearson correlation in mean mood across sessions: {corr_meanmood}, P value: {p_meanmood}")

# Mean Mood ICC

# Convert the wide-format mood table into long format for ICC analysis
MeanMood_long = MeanMood.melt(id_vars='participant_id',
                  value_vars=['mean_ratings_1', 'mean_ratings_2'],
                  var_name='session', value_name='mean_mood')
MeanMood_long['session'] = MeanMood_long['session'].str.extract(r'_(\d)').astype(int)

# Compute ICC
MeanMood_iccresult = pg.intraclass_corr(data=MeanMood_long, targets='participant_id', raters='session', ratings='mean_mood')
print("\nMean Mood ICC:")
print(MeanMood_iccresult[['Type', 'ICC', 'CI95%', 'pval']])

# Correlation in mood range across sessions

first_MinMax = first_df.groupby('participant_id')['happySlider.response'].agg(
    first_minmood = 'min',
    first_maxmood = 'max',).reset_index()

first_MinMax['range_1'] = first_MinMax['first_maxmood'] - first_MinMax['first_minmood'] 

second_MinMax = second_df.groupby('participant_id')['happySlider.response'].agg(
    second_minmood = 'min',
    second_maxmood = 'max',).reset_index()

second_MinMax['range_2'] = second_MinMax['second_maxmood'] - second_MinMax['second_minmood']

RangeMood = pd.merge(first_MinMax, second_MinMax, on='participant_id')

# Pearson correlation
corr_rangemood, p_rangemood = stats.pearsonr(RangeMood['range_1'], RangeMood['range_2'])
print(f"\nPearson correlation in mood range across sessions: {corr_rangemood}, P value: {p_rangemood}")

# Mood Range ICC

RangeMood_long = RangeMood.melt(id_vars='participant_id',
                  value_vars=['range_1', 'range_2'],
                  var_name='session', value_name='mood_range')
RangeMood_long['session'] = RangeMood_long['session'].str.extract(r'_(\d)').astype(int)

# Compute ICC
RangeMood_iccresult = pg.intraclass_corr(data=RangeMood_long, targets='participant_id', raters='session', ratings='mood_range')
print("\nMood Range ICC:")
print(RangeMood_iccresult[['Type', 'ICC', 'CI95%', 'pval']])

# Correlation between the ratings standard deviation and the first mood rating (from the first seesion)

first_rating1 = first_df.groupby('participant_id')['happySlider.response'].first().reset_index(name='first_rating')
first_ratingstd = first_df.groupby('participant_id')['happySlider.response'].std().reset_index(name='ratings_std')
first_rating1_std_df = first_rating1.merge(first_ratingstd, on='participant_id')

# Spearman correlation
first_corr_rating1std, first_p_rating1std = stats.spearmanr(first_rating1_std_df['first_rating'], first_rating1_std_df['ratings_std'])
print(f"\nSpearman correlation between ratings std and the first mood rating: {first_corr_rating1std}, P value: {first_p_rating1std}")


#%% Model Results analysis

# Correlation in betaR across sessions

# Pearson correlation
corr_betaR, p_betaR = stats.pearsonr(betaR_df['betaR_1'], betaR_df['betaR_2'])
print(f"\nPearson correlation in betaR across sessions: {corr_betaR}, P value: {p_betaR}")

# betaR ICC

betaR_long = betaR_df.melt(id_vars='participant_id',
                  value_vars=['betaR_1', 'betaR_2'],
                  var_name='session', value_name='betaR')
betaR_long['session'] = betaR_long['session'].str.extract(r'_(\d)').astype(int)

# Compute ICC
betaR_iccresult = pg.intraclass_corr(data=betaR_long, targets='participant_id', raters='session', ratings='betaR')
print("\nbetaR ICC:")
print(betaR_iccresult[['Type', 'ICC', 'CI95%', 'pval']])

# Correlation between betaR and the mood range

RangeMood = RangeMood.merge(betaR_df, on='participant_id')

# Pearson correlation- Session 1
first_corr_ranger, first_p_ranger = stats.pearsonr(RangeMood['range_1'], RangeMood['betaR_1'])
print(f"\nPearson correlation between mood range session 1 and betaR: {first_corr_ranger}, P value: {first_p_ranger}")

# Pearson correlation- Session 2
second_corr_ranger, second_p_ranger = stats.pearsonr(RangeMood['range_2'], RangeMood['betaR_2'])
print(f"\nPearson correlation between mood range session 2 and betaR: {second_corr_ranger}, P value: {second_p_ranger}")

# Correlation between betaR and the first mood rating (from the first seesion)

first_rating1 = first_df.groupby('participant_id')['happySlider.response'].first().reset_index(name='first_rating')
first_rating_df = betaR_df.merge(first_rating1, on='participant_id')

# Spearman correlation
first_corr_rating1r, first_p_rating1r = stats.spearmanr(first_rating_df['first_rating'], first_rating_df['betaR_1'])
print(f"\nSpearman correlation between the first rating and betaR: {first_corr_rating1r}, P value: {first_p_rating1r}")

# Correlation between betaR and mood consistency

correlation_results = []

for participant in first_df['participant_id'].unique():
    
    # Get mood ratings for each session, drop NaNs
    first = first_df[first_df['participant_id'] == participant]['happySlider.response'].dropna().reset_index(drop=True)
    second = second_df[second_df['participant_id'] == participant]['happySlider.response'].dropna().reset_index(drop=True)
    
    r_trajectories, p_trajectories = stats.pearsonr(first, second) # Compute Pearson correlation between Session 1 and Session 2 mood trajectories

    correlation_results.append({
        'participant_id': participant,
        'r_pearson': r_trajectories,
        'p_value': p_trajectories,
    })

consistency_df = pd.DataFrame(correlation_results)
consistency_betaR = consistency_df.merge(betaR_df, on='participant_id')

# Pearson correlation
corr_consR, p_consR = stats.pearsonr(consistency_betaR['r_pearson'], consistency_betaR['betaR_1'])
print(f"\nPearson correlation between mood consistency and betaR: {corr_consR}, P value: {p_consR}")

# Correlation between betaR-betaP (from the first seesion) and mood range consistency

first_modelresults = modelresults_df[modelresults_df['round']==1]
first_modelresults['betas_diff'] = first_modelresults['betaR'] - first_modelresults['betaP']

# Compute within-participant consistency of mood range
RangeMood['mood_range_consistency'] = (
    np.minimum(RangeMood['range_1'], RangeMood['range_2']) /
    np.maximum(RangeMood['range_1'], RangeMood['range_2']))
RangeMood.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace divisions by zero or invalid values with NaN

range_consistency_df = RangeMood[['participant_id', 'mood_range_consistency']].merge(first_modelresults[['participant_id', 'betas_diff']], on = 'participant_id')

# Pearson correlation
corr_rcons_betas, p_rcons_betas = stats.pearsonr(range_consistency_df['mood_range_consistency'], range_consistency_df['betas_diff'])
print(f"\nPearson correlation between betaR-betaP (from the first seesion) and mood range consistency: {corr_rcons_betas}, P value: {p_rcons_betas}")

# Correlation in betaP across sessions

# Pearson correlation
corr_betaP, p_betaP = stats.pearsonr(betaP_df['betaP_1'], betaP_df['betaP_2'])
print(f"\nPearson correlation in betaP across sessions: {corr_betaP}, P value: {p_betaP}")

# betaP ICC
betaP_long = betaP_df.melt(id_vars='participant_id',
                  value_vars=['betaP_1', 'betaP_2'],
                  var_name='session', value_name='betaP')
betaP_long['session'] = betaP_long['session'].str.extract(r'_(\d)').astype(int)

# Compute ICC
betaP_iccresult = pg.intraclass_corr(data=betaP_long, targets='participant_id', raters='session', ratings='betaP')
print("\nbetaP ICC:")
print(betaP_iccresult[['Type', 'ICC', 'CI95%', 'pval']])

# Correlation between betaP and the mood range
RangeMood = RangeMood.merge(betaP_df, on='participant_id')

# Pearson correlation- Session 1
first_corr_rangeP, first_p_rangeP = stats.pearsonr(RangeMood['range_1'], RangeMood['betaP_1'])
print(f"\nPearson correlation between mood range session 1 and betaP: {first_corr_rangeP}, P value: {first_p_rangeP}")

# Pearson correlation- Session 2
second_corr_rangeP, second_p_rangeP = stats.pearsonr(RangeMood['range_2'], RangeMood['betaP_2'])
print(f"\nPearson correlation between mood range session 2 and betaP: {second_corr_rangeP}, P value: {second_p_rangeP}")


# Correlation between betP and mood consistency
correlation_results = []

for participant in first_df['participant_id'].unique():
    
    # Get mood ratings for each session, drop NaNs
    first = first_df[first_df['participant_id'] == participant]['happySlider.response'].dropna().reset_index(drop=True)
    second = second_df[second_df['participant_id'] == participant]['happySlider.response'].dropna().reset_index(drop=True)
    
    r_trajectories, p_trajectories = stats.pearsonr(first, second) # Compute Pearson correlation between Session 1 and Session 2 mood trajectories

    correlation_results.append({
        'participant_id': participant,
        'r_pearson': r_trajectories,
        'p_value': p_trajectories,
    })

consistency_df = pd.DataFrame(correlation_results)
consistency_betaP = consistency_df.merge(betaP_df, on='participant_id')

# Pearson correlation
corr_consP, p_consP = stats.pearsonr(consistency_betaP['r_pearson'], consistency_betaP['betaP_1'])
print(f"\nPearson correlation between mood consistency and betaP: {corr_consP}, P value: {p_consP}")

# Spearman correlation
corr_consP, p_consP = stats.spearmanr(consistency_betaP['r_pearson'], consistency_betaP['betaP_1'])
print(f"\nSpearman correlation between mood consistency and betaP: {corr_consP}, P value: {p_consP}")







