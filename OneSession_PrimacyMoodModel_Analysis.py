# -*- coding: utf-8 -*-
"""
Article Code: "Dynamic emotional updating as a computational marker of well-being"
@Code Author: Noa Nagar


Code 3- One Session Primacy Mood Model Analysis

The Primacy Mood Model, his implementation with Statistical analyses correlations (and regression model for betaR) for each condition in the one session experiment. 

First, run code 1 ('OneSession_Preprocessing') or download the 'HighMoodCon_preprocessed.csv' and 'LowMoodCon_preprocessed.csv' files from the 'onseSessionData' folder.
Then, download the files 'HighMoodCon_ParticipantSummary.xlsx' and 'LowMoodCon_ParticipantSummary.xlsx' from the 'onseSessionData' folder.
finally run this code.

 
Here we show the anaysis without the z-score normalization.

"""

#%% Import Libraries

import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, Bounds
from statsmodels.formula.api import ols

#%% Import Data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "oneSessionData")

# High Mood Target
HighMoodCon_DIR = os.path.join(DATA_DIR, "HighMoodCon_preprocessed.csv")
high_df = pd.read_csv(HighMoodCon_DIR)

HighSummary_DIR = os.path.join(DATA_DIR, "HighMoodCon_ParticipantSummary.xlsx")
high_summary_df = pd.read_excel(HighSummary_DIR)

# Low Mood Target
LowMoodCon_DIR = os.path.join(DATA_DIR, "LowMoodCon_preprocessed.csv")
low_df = pd.read_csv(LowMoodCon_DIR)

LowSummary_DIR = os.path.join(DATA_DIR, "LowMoodCon_ParticipantSummary.xlsx")
low_summary_df = pd.read_excel(LowSummary_DIR)


#%% The Primacy Mood Model

class CurveLTA(object):
    
    # Class Attributes
    _per_names = ['m0', 'lam', 'betaP', 'betaR']
    _default_pars = [0.5, 0.8, 0.01, 0.005] 
    
    # Constraints for parameter optimization:
    _lower_bounds = [0.0, 0.0, 1e-8, 1e-8] 
    _upper_bounds = [1.0, 1.0, np.inf, np.inf]
      
    # Penalty terms for regularization
    pen_betaP = 0.0
    pen_betaR = 0.0
    _model_name = 'LTA with nonlinear utilities'

    def __init__ (self, pen_betaP=0.0, pen_betaR=0.0):
        self.pen_betaP = pen_betaP
        self.pen_betaR = pen_betaR
    
    def __str__(self):
        """
        Return a human-readable string representation of the model.
        Lists the model name and current values of all parameters.
        """
        s = self._model_name
        for par in self._par_names:
          s = s + '\n' + par + ': %.4f' % self.__dict__[par] 
    
        return s
    
    # Initialize Default Parameters
    def initialize(self):
        """
        Initialize the model parameters to their default values.
        """
        self.m0, self.lam, self.betaP, self.betaR = self._default_pars
    
    # Model Fitting
    def fit(self, actual, mood):
        """
        Fit the model parameters to the provided data.
    
        Inputs: actual= Outcomes; timestamps= Time indices; mood= Observed mood ratings.
        """
        def loss_func(par):
            """
            Calculate the loss between the model's predicted mood and the actual observed mood, while incorporating penalties for the model parameters.
            Parameters: par= A list of parameters to optimize.
            """
            self.m0, self.lam, self.betaP, self.betaR = par # Assigns current parameter values to the model
            mood_pred = self.predict(actual)
            pen_term = np.abs(self.pen_betaP * self.betaP) + \
                np.abs(self.pen_betaR * self.betaR)  # Computes a penalty term based on regularization parameters.
            loss = np.nansum(np.abs(mood - mood_pred)) + pen_term 
    
            return loss
    
        # Optimization
        res = minimize(loss_func, self._default_pars, bounds=Bounds(self._lower_bounds, self._upper_bounds)) # optimize the loss_func. 
        self.m0, self.lam, self.betaP, self.betaR = res.x # Updates the model parameters with the optimized values
    
        return res
    
    # Mood Prediction
    def predict(self, actual):
        n_trials = len(actual)
        mood_pred = np.zeros(n_trials) # Store mood predictions.
    
        # Holds the exponentially weighted sums for P(t) and R(t)
        sum_P = 0 # expectations
        sum_R = 0 # actual outcomes
    
        for trial_no in range(n_trials):
            if trial_no == 0:
                lte = 0 
            else:
                lte = np.mean(actual[:trial_no]) # Expectation for subsequent trials= average of previous outcomes.
    
            # Update sum_P and sum_R using the exponential decay factor
            sum_P = sum_P * self.lam + lte
            sum_R = sum_R * self.lam + actual[trial_no]
        
            # Mood Prediction
            mood_mu = self.m0 + self.betaP * sum_P + self.betaR * sum_R
            mood_pred[trial_no] = mood_mu
    
        return mood_pred

#%%
def model_results(model_data):
    # Create Empty DataFrame
    results_df = pd.DataFrame(columns = ['participant','CESD10_score','Group', 'm0', 'lam', 'betaP', 'betaR', 'winAmountAVG', 'loseAmountAVG', 'MSE']) 
    mood_df =  pd.DataFrame(columns = ['participant', 'trial_index', 'mood_prediction', 'actual_mood'])

    # Group Data by Participant
    grouped = model_data.groupby('participant')

    for participant, group in grouped:
        clean_group = group.dropna(subset = ['happySlider.response']) # Clean Missing Mood Data
        # Skip Empty Groups
        if clean_group.empty:
            print(f"No valid mood data for participant {participant}")
            continue

        # Prepare Data for the Model- Defining variables
        mood = clean_group['happySlider.response'].values
        CESD_score = clean_group['CESD10_Score'].iloc[0]
        CESD_group = clean_group['Group'].iloc[0]
        outcomes = clean_group['outcome'].values
        wamt = clean_group['winAmount'].values
        lamt = clean_group['loseAmount'].values
        camt = clean_group['certainAmount'].values
        wins = (outcomes == 'win').astype(int)
        gambels = (clean_group['choice'] == 'gamble').values
        currentAmount = clean_group['outcomeAmount'].values
    
        # Initialize and Fit the Model
        model = CurveLTA(pen_betaP = 0.1, pen_betaR = 0.1)
        model.fit(currentAmount, mood)
    
        # Predict Mood
        mood_pred = model.predict(currentAmount)
    
        # Calculate Mean Squared Error (MSE)
        mse = np.mean((mood - mood_pred) ** 2)
    
        # Add Results to DataFrame
        results_df = pd.concat([results_df, pd.DataFrame([{
          'participant': participant,
          'CESD10_score': CESD_score.astype(float),
          'Group': CESD_group,
          'm0': model.m0,
          'lam': model.lam,
          'betaP': model.betaP,
          'betaR': model.betaR,
          'winAmountAVG': clean_group['winAmount'].mean(),
          'loseAmountAVG': clean_group['loseAmount'].mean(),
          'MSE': mse
      }])], ignore_index=True)
    
        trial_preds = pd.DataFrame({
                        'participant': participant,
                        'trial_index': clean_group.index,
                        'actual_mood': mood,
                        'mood_prediction': mood_pred
                    })
        mood_df = pd.concat([mood_df, trial_preds], ignore_index=True)
    
    return results_df, mood_df

# High Mood Target
HighCon_ModelResults, HighCon_mood = model_results(high_df)
HighCon_ALLResults = pd.merge(high_summary_df, HighCon_ModelResults, on= 'participant', how= 'left')

higt_output_path = os.path.join(DATA_DIR, "HighCon_ALLResults.csv")
HighCon_ALLResults.to_csv(higt_output_path, index=False)

# Low Mood Target
LowCon_ModelResults, LowCon_mood = model_results(low_df)
LowCon_ALLResults = pd.merge(low_summary_df, LowCon_ModelResults, on= 'participant', how= 'left')

low_output_path = os.path.join(DATA_DIR, "LowCon_ALLResults.csv")
LowCon_ALLResults.to_csv(low_output_path, index=False)

#%% Correlations

#%% Correlations

# Correlation between betaR and the standart deviation of the mood ratings
def stdmood_r_correlation(df):
    
    # Pearson correlation
    Pcorr, Pp_value = stats.pearsonr(df['betaR'], df['Ratings_ST'])
    print(f"\nPearson correlation between betaR and the standart deviation of the mood ratings: {Pcorr}, P value: {Pp_value}")
    
    # Spearman Correlations
    Scorr, Sp_value = stats.spearmanr(df['betaR'], df['Ratings_ST'])
    print(f"\nSpearman correlation between betaR and the standard deviation of the mood ratings: {Scorr}, P value: {Sp_value}")
    
    return Pcorr, Pp_value, Scorr, Sp_value

# High Mood Target
print("\nHigh Mood Target")
high_Pcorr_stdr, high_Pp_stdr, high_Scorr_stdr, high_Sp_stdr= stdmood_r_correlation(HighCon_ALLResults) 

# Low Mood Target
print("\nLow Mood Target")
low_Pcorr_stdr, low_Pp_stdr, low_Scorr_stdr, low_Sp_stdr = stdmood_r_correlation(LowCon_ALLResults) 

# Correlation between betaR and the range of the mood ratings
def rangemood_r_correlation(df):
    
    # Pearson correlation
    Pcorr, Pp_value = stats.pearsonr(df['betaR'], df['range'])
    print(f"\nPearson correlation between betaR and the range of the mood ratings: {Pcorr}, P value: {Pp_value}")
    
    # Spearman Correlations
    Scorr, Sp_value = stats.spearmanr(df['betaR'], df['range'])
    print(f"\nSpearman correlation between betaR and the range of the mood ratings: {Scorr}, P value: {Sp_value}")
    
    return Pcorr, Pp_value, Scorr, Sp_value

# High Mood Target
print("\nHigh Mood Target")
high_Pcorr_ranger, high_Pp_ranger, high_Scorr_ranger, high_Sp_ranger = rangemood_r_correlation(HighCon_ALLResults) 

# Low Mood Target
print("\nLow Mood Target")
low_Pcorr_ranger, low_Pp_ranger, low_Scorr_ranger, low_Sp_ranger = rangemood_r_correlation(LowCon_ALLResults)

def stdmood_rp_correlation(df):
    
    df['beta_diff'] = df['betaR'] - df['betaP']
    
    # Pearson correlation
    Pcorr, Pp_value = stats.pearsonr(df['beta_diff'], df['Ratings_ST'])
    print(f"\nPearson correlation between betaR - betaP and the standart deviation of the mood ratings: {Pcorr}, P value: {Pp_value}")
    
    # Spearman Correlations
    Scorr, Sp_value = stats.spearmanr(df['beta_diff'], df['Ratings_ST'])
    print(f"\nSpearman correlation between betaR - betaP and CESD-10 score: {Scorr}, P value: {Sp_value}")
    
    
    return Pcorr, Pp_value, Scorr, Sp_value

# High Mood Target
print("\nHigh Mood Target")
high_Pcorr_stdrp, high_Pp_stdrp, high_Scorr_stdrp, high_Sp_stdrp = stdmood_rp_correlation(HighCon_ALLResults) 

# Correlation between betaR - betaP and CESD-10 score
def cesd_rp_correlation(df):
    
    df['beta_diff'] = df['betaR'] - df['betaP']
    
    # Pearson correlation
    Pcorr, Pp_value = stats.pearsonr(df['beta_diff'], df['CESD10_score'])
    print(f"\nPearson correlation between betaR - betaP and CESD-10 score: {Pcorr}, P value: {Pp_value}")
    
    # Spearman Correlations
    Scorr, Sp_value = stats.spearmanr(df['beta_diff'], df['CESD10_score'])
    print(f"\nSpearman correlation between betaR - betaP and CESD-10 score: {Scorr}, P value: {Sp_value}")
    
    return Pcorr, Pp_value, Scorr, Sp_value

# High Mood Target
print("\nHigh Mood Target")
high_Pcorr_cesdrp, high_Pp_cesdrp, high_Scorr_cesdrp, high_Sp_cesdrp = cesd_rp_correlation(HighCon_ALLResults)

# Low Mood Target
print("\nLow Mood Target")
low_Pcorr_cesdrp, low_Pp_cesdrp, low_Scorr_cesdrp, low_Sp_cesdrp = cesd_rp_correlation(LowCon_ALLResults) 

# Correlation between betaR and CESD-10 score
def cesd_r_correlation(df):
  
    # Pearson correlation
    Pcorr, Pp_value = stats.pearsonr(df['betaR'], df['CESD10_score'])
    print(f"\nPearson correlation between betaR and CESD-10 score: {Pcorr}, P value: {Pp_value}")
    
    # Spearman Correlations
    Scorr, Sp_value = stats.spearmanr(df['betaR'], df['CESD10_score'])
    print(f"\nSpearman correlation between betaR and CESD-10 score: {Scorr}, P value: {Sp_value}")
    
    return Pcorr, Pp_value, Scorr, Sp_value

# High Mood Target
print("\nHigh Mood Target")
high_Pcorr_cesdr, high_Pp_cesdr, high_Scorr_cesdr, high_Sp_cesdr = cesd_r_correlation(HighCon_ALLResults) 

# Correlation between betaP and CESD-10 score
def cesd_p_correlation(df):
  
    # Pearson correlation
    Pcorr, Pp_value = stats.pearsonr(df['betaP'], df['CESD10_score'])
    print(f"\nPearson correlation between betaP and CESD-10 score: {Pcorr}, P value: {Pp_value}")
    
    # Spearman Correlations
    Scorr, Sp_value = stats.spearmanr(df['betaP'], df['CESD10_score'])
    print(f"\nSpearman correlation between betaP and CESD-10 score: {Scorr}, P value: {Sp_value}")
    
    return Pcorr, Pp_value, Scorr, Sp_value

# High Mood Target
print("\nHigh Mood Target")
high_Pcorr_cesdp, high_Pp_cesdp, high_Scorr_cesdp, high_Sp_cesdp = cesd_p_correlation(HighCon_ALLResults) 

#%% Regression Model

# First rating
high_model_cesdR_First = ols('CESD10_score ~ betaR + Ratings_Mean + range + Rating_first', data = HighCon_ALLResults).fit()
print("\nHigh Mood Model: CESD10_score ~ betaR + Ratings_Mean + range + Rating_first")
print(high_model_cesdR_First.summary())

high_model_cesdP_First = ols('CESD10_score ~ betaP + Ratings_Mean + range + Rating_first', data = HighCon_ALLResults).fit()
print("\nHigh Mood Model: CESD10_score ~ betaP + Ratings_Mean + range + Rating_first")
print(high_model_cesdP_First.summary())

#Target dis
HighCon_ALLResults['TragetDis'] = (0.85- HighCon_ALLResults['Rating_first']).abs()
LowCon_ALLResults['TragetDis'] = (0.30- LowCon_ALLResults['Rating_first']).abs()

high_model_cesdR_TargetDis = ols('CESD10_score ~ betaR + Ratings_Mean + range + TragetDis', data = HighCon_ALLResults).fit()
print("\nHigh Mood Model: CESD10_score ~ betaR + Ratings_Mean + range + TragetDis")
print(high_model_cesdR_TargetDis.summary())

high_model_cesdP_TargetDis = ols('CESD10_score ~ betaP + Ratings_Mean + range + TragetDis', data = HighCon_ALLResults).fit()
print("\nHigh Mood Model: CESD10_score ~ betaP + Ratings_Mean + range + TragetDis")
print(high_model_cesdP_TargetDis.summary())

# CESD X All conditions Model
LowCon_ALLResults["Target"] = 0
HighCon_ALLResults["Target"] = 1

allCon = pd.concat([LowCon_ALLResults, HighCon_ALLResults], ignore_index=True)

allCon = allCon.dropna(subset=["betaR", "CESD10_score", "Target"]).copy()

allCon["CESD_centered"] = (allCon["CESD10_score"] - allCon["CESD10_score"].mean())
allCon_model = ols("betaR ~ CESD_centered * Target", data=allCon).fit()

print("\nModel: betaR ~ CESD_centered * Target")
print(allCon_model.summary())