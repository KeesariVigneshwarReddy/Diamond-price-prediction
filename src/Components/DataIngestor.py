import os
import sys
import warnings
warnings.filterwarnings('ignore')
from src.Exception import CustomException
from src.Logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass
class DataIngestionConfig :
    raw_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "01_raw.csv")
    raw_dropped_renamed_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "02_df_id_target.csv")
    X_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "03_X.csv")
    y_data_path : str = os.path.join(r"C:/ML Projects/Diamond Price Prediction/Artifacts/Modular intermediate datasets", "04_y.csv")

class DataIngestion :
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try :
            
            # Load data set
            df = pd.read_csv(r"C:/ML Projects/Diamond Price Prediction/Notebooks/Data/gemstone.csv")
            logging.info('Read the dataset as dataframe')
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            # Drop duplicated and rename columns
            df.drop_duplicates(inplace = True)
            logging.info('Duplicate datapoints are dropped')
            df.drop(columns = ['id'], inplace = True)
            logging.info("'id' column is dropped")
            df.rename(columns = {'price': 'target'}, inplace = True)
            logging.info("'price' column is renamed to 'target'")
            df.to_csv(self.ingestion_config.raw_dropped_renamed_data_path, index = False, header = True)

            # Split the data frame in to X, y
            X = df.drop(columns = ['target'])
            y = df['target']
            logging.info('Dataset is splitted into X, y')
            X.to_csv(self.ingestion_config.X_data_path, index = False, header = True)
            y.to_csv(self.ingestion_config.y_data_path, index = False, header = True)


            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.raw_dropped_renamed_data_path,
                self.ingestion_config.X_data_path,
                self.ingestion_config.y_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

"""
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

"""