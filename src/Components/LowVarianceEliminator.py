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
from sklearn.feature_selection import VarianceThreshold

@dataclass
class LowVarianceEliminatorConfig :
    
    X_removed_low_variance_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_removed_low_variance.csv")
   
class LowVarianceEliminator :
    def __init__(self):
        self.low_variance_eliminator_config = LowVarianceEliminatorConfig()
        
    
    def initiate_low_variance_eliminator(self, X_data_path) :
        logging.info("Entered the Low Variance Eliminator component")
        try :
            
            # Load X
            X = pd.read_csv(X_data_path)
            logging.info('Read X as dataframe')

            # Drop low variance features - Determine threshold according to your use case
            var_thres_selector = VarianceThreshold(threshold = 0.0000001)
            var_thres_selector.fit_transform(X)
            constant_columns = [column for column in X.columns
                                if column not in X.columns[var_thres_selector.get_support()]]
            X.drop(constant_columns, axis = 1, inplace = True)
            X.drop(constant_columns, axis = 1, inplace = True)

            X.to_csv(self.low_variance_eliminator_config.X_removed_low_variance_data_path, index = False, header = True)
            logging.info("Low variance columns are removed")

            return  (
                        self.low_variance_eliminator_config.X_removed_low_variance_data_path,
                        constant_columns
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_low_varaince_eliminator")

            raise CustomException(e, sys)