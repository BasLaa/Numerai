# Numerai
Some of my experiments and models for the Numerai tournament.

## Files
The **Optuna_Cross_Val** file gives a template for doing hyperparameter search using the Optuna framework. This is a great tool, as you can visualize all the trials nicely, and it employs a search based on the TPE (Tree-structured Parzen Estimator) algorithm.

**Metrics.py** gives a function which calculates the correlation, sharpe and max feature exposure for your predictions.

**era_comparison** is a notebook I wrote while doing some experiments on eras. It splits the eras into groups (50, then 15), and trains a linear model on each group, testing on all the others. I did the same with a LightGBM model. You can see some interesting patterns in the visualization, suggesting that some eras are more similar to others. I'm currently working on training on different era groups and ensembling.

**boost_training** uses the era boosting algorithm provided by the team to train on the train data and predict on the current tournament round. It uses some hyperparameters I found with Optuna.
