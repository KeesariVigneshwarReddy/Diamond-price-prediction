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
from sklearn.model_selection import train_test_split

@dataclass
class TrainTestSplitterConfig :
    
    X_train_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_train.csv")
    y_train_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "y_train.csv")
    X_test_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X_test.csv")
    y_test_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "y_test.csv")

class TrainTestSplitter :
    def __init__(self):
        self.train_test_splitter_config = TrainTestSplitterConfig()
        
    def initiate_train_test_splitter(self, X_data_path, y_data_path) :
        logging.info("Entered the Train Test Splitter component")
        try :
            
            # Load X and y
            X = pd.read_csv(X_data_path)
            y = pd.read_csv(y_data_path)
            logging.info('Read X and y as dataframe')

            # Train test split
            # Split ratio train : test
            split_ratio = [80, 20]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = float(split_ratio[1] / 100), random_state = 42)
            X_train.reset_index(drop = True, inplace = True)
            y_train.reset_index(drop = True, inplace = True)
            X_test.reset_index(drop = True, inplace = True)
            y_test.reset_index(drop = True, inplace = True)
            X_train.to_csv(self.train_test_splitter_config.X_train_data_path, index = False, header = True)
            y_train.to_csv(self.train_test_splitter_config.y_train_data_path, index = False, header = True)
            X_test.to_csv(self.train_test_splitter_config.X_test_data_path, index = False, header = True)
            y_test.to_csv(self.train_test_splitter_config.y_test_data_path, index = False, header = True)

            logging.info('Train test split completed')

            return (
                        self.train_test_splitter_config.X_train_data_path,
                        self.train_test_splitter_config.y_train_data_path,
                        self.train_test_splitter_config.X_test_data_path,
                        self.train_test_splitter_config.y_test_data_path
                    )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)