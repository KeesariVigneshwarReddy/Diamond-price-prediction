import os
import sys
import copy
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from src.Exception import CustomException
from src.Logger import logging

import joblib
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

@dataclass
class CorrelationEliminatorConfig :
    
    X_MIF_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_MIF.csv")

class CorrelationEliminator :
    def __init__(self):
        self.correlation_eliminator_config = CorrelationEliminatorConfig()
        

    def initiate_correlation_eliminator(self, X_data_path, y_data_path) :
        logging.info("Entered the Correlation Eliminator component")
        try :
            
            # Load X, y
            X = pd.read_csv(X_data_path)
            y = pd.read_csv(y_data_path)
            logging.info('Read X, y as dataframe')

            # Creating X_y 
            X_y = pd.concat([X, y], axis = 1)
            logging.info('X_y has been created')

            spearman_corr_matrix_target, p_values = spearmanr(X_y)
            spearman_corr_df_target = pd.DataFrame(spearman_corr_matrix_target, columns = X_y.columns, index = X_y.columns)

            spearman_corr_matrix, p_values = spearmanr(X)
            spearman_corr_df = pd.DataFrame(spearman_corr_matrix, columns = X.columns, index = X.columns)

            # Correlation of features on the basis of spearman correlation coefficient
            threshold_min = 0.1 # min correlation with target needed to be selected

             # Removing the features with are least correlated with target
            features_not_correlated_with_target = [] 

            for i in range(len(spearman_corr_df_target.columns)) :
                if abs(spearman_corr_df_target.iloc[i, len(spearman_corr_df_target.columns) - 1]) < threshold_min :
                    features_not_correlated_with_target.append(spearman_corr_df_target.columns[i])

            # Removing the features with are highly correlated to each other
            threshold_max = 0.85 # max correlation needed to be selected

            features_correlated = {} 

            for i in range(len(spearman_corr_df.columns)) :
                for j in range(len(spearman_corr_df.columns)) :
                    if i == j :
                        continue
                    if spearman_corr_df.iloc[i, j] > threshold_max :
                        list_key = features_correlated.get(spearman_corr_df.columns[i], [])
                        list_key.append(spearman_corr_df.index[j])
                        features_correlated[spearman_corr_df.columns[i]] = list_key

            features_to_be_dropped = copy.deepcopy(features_not_correlated_with_target) + ['x', 'y', 'z'] # Features to be dropped  
                                                                                                          # if they are highly correlated. Select from EDA or features_correlated        
            
            X.drop(columns = features_to_be_dropped, inplace = True)
            X.to_csv(self.correlation_eliminator_config.X_MIF_data_path, index = False, header = True)
            logging.info("Features are removed on the basis of correlation")

            return (
                        self.correlation_eliminator_config.X_MIF_data_path,
                        features_to_be_dropped
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_correlation_eliminator")

            raise CustomException(e, sys)