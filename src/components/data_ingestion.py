from dataclasses import dataclass
import os
import pandas as pd
import sklearn
import sys
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataIngestionConfig():
    train_data_path = os.path.join("data","train.csv")
    test_data_path = os.path.join("data","test.csv")
    raw_data_path = os.path.join("data", "data")

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv("")
            logging.info("Read the dataset")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            df_train,df_test = train_test_split(df, test_size=0.2)
            df_train.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            df_test.to_csv(self.ingestion_config.test_data_path, index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)
