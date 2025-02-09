import os
import sys
import json

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from superstoresales.exception.exception import SuperStoreSalesException
from superstoresales.logging.logger import logging

class SuperStoreSalesDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def csv_to_json_converter(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        
        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        

if __name__ =="__main__":
    FILE_PATH = "sales_data/sales_data.csv"
    DATABASE = "AP-DB"
    collection = "SuperStoreSalesData"
    superstoresalesobj = SuperStoreSalesDataExtract()
    records = superstoresalesobj.csv_to_json_converter(file_path=FILE_PATH)
    print(records)
    no_of_records = superstoresalesobj.insert_data_mongodb(records=records,database=DATABASE,collection=collection)
    print(no_of_records)