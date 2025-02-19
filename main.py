from superstoresales.components.data_ingestion import DataIngestion

from superstoresales.exception.exception import SuperStoreSalesException
from superstoresales.logging.logger import logging

from superstoresales.entity.config_entity import DataIngestionConfig
from superstoresales.entity.config_entity import TrainingPipelineConfig

import sys


if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()

        # data ingestion
        dataingestionconfig = DataIngestionConfig(training_pipeline_config=trainingpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config=dataingestionconfig)
        logging.info("Initiate the data ingestion process.")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed")
        print(dataingestionartifact)
    except Exception as e:
        raise SuperStoreSalesException(e,sys)