from scipy.stats import spearmanr
import pandas as pd
import numpy as np

def evaluate(df, pred_cols, verbose=True, fast=False):
    """
    Evaluate and display relevant metrics for Numerai 
    
    :param df: A Pandas DataFrame containing the columns "era", "target" and "prediction"
    :return: A tuple of float containing the metrics
    """
    validation_stats = pd.DataFrame()

    def _score(sub_df: pd.DataFrame, pred_col) -> np.float32:
        """Calculates Spearman correlation"""
        return spearmanr(sub_df["target_nomi_20"], sub_df[pred_col])[0]

    def feature_exposures(df: pd.DataFrame, pred_col):
      feature_names = [f for f in df.columns
                      if f.startswith("feature")]
      exposures = []
      for f in feature_names:
          fe = spearmanr(df[pred_col], df[f])[0]
          exposures.append(fe)
      return np.array(exposures)

    def max_feature_exposure(df, pred_col):
        return np.max(np.abs(feature_exposures(df, pred_col)))


    def feature_exposure(df, pred_col):
        return np.sqrt(np.mean(np.square(feature_exposures(df, pred_col))))
    
    for pred_col in pred_cols:

      # Calculate metrics
      corrs = df.groupby("era").apply(_score, pred_col)
      spearman = round(corrs.mean(), 4)
      numerai_sharpe = round(corrs.mean() / corrs.std(), 4)
      if not fast:
        max_exposure = round(max_feature_exposure(df, pred_col), 4)
      else:
        max_exposure = 0
      
      validation_stats.loc["spearman", pred_col] = spearman
      validation_stats.loc["sharpe", pred_col] = numerai_sharpe
      validation_stats.loc["max_feature_exposure", pred_col] = max_exposure

      if verbose:
        # Display metrics
        print(f"Metrics for {pred_col}")
        print(f"Spearman Correlation: {spearman}")
        print(f"Sharpe Ratio: {numerai_sharpe}")
        print(f"Max Feature Exposure:  {max_exposure}")

    return validation_stats