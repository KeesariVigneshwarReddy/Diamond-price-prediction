import os
import sys
import copy
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass

from src.Exception import CustomException
from src.Logger import logging

import pandas as pd

@dataclass
class DataIngestorConfig :
    raw_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "raw.csv")
    raw_dropped_renamed_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "df_id_target.csv")
    X_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "X.csv")
    y_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "y.csv")

class DataIngestor :
    def __init__(self):
        self.data_ingestor_config = DataIngestorConfig()

    def initiate_data_ingestor(self):
        logging.info("Entered the Data Ingestor component")
        try :
            
            # Load data set
            df = pd.read_csv(r"C:/ML Projects/Diamond Price Prediction/Notebooks/Data/gemstone.csv")
            logging.info('Read the dataset as dataframe')
            df.to_csv(self.data_ingestor_config.raw_data_path, index = False, header = True)
            logging.info("Raw dataset is saved")

            # Drop duplicated and rename columns
            df.drop_duplicates(inplace = True)
            df.drop(columns = ['id'], inplace = True)
            df.rename(columns = {'price': 'target'}, inplace = True)
            df.to_csv(self.data_ingestor_config.raw_dropped_renamed_data_path, index = False, header = True)
            logging.info("Duplicate datapoints are dropped, 'id' column is dropped, 'price' column is renamed to 'target'")

            # Split the data frame in to X, y
            X = df.drop(columns = ['target'])
            y = df['target']
            X.to_csv(self.data_ingestor_config.X_data_path, index = False, header = True)
            y.to_csv(self.data_ingestor_config.y_data_path, index = False, header = True)
            logging.info('Dataset is splitted into X, y')


            logging.info("Data ingestion is completed")

            return(
                self.data_ingestor_config.raw_data_path,
                self.data_ingestor_config.raw_dropped_renamed_data_path,
                self.data_ingestor_config.X_data_path,
                self.data_ingestor_config.y_data_path
            )
        
        except Exception as e :
            logging.info('Exception occured in initiate_data_ingestor')
            raise CustomException(e, sys)