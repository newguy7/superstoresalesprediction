from superstoresales.components.data_ingestion import DataIngestion
from superstoresales.components.data_validation import DataValidation

from superstoresales.exception.exception import SuperStoreSalesException
from superstoresales.logging.logger import logging

from superstoresales.entity.config_entity import DataIngestionConfig, DataValidationConfig
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

        # data validation
        data_validation_config = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        data_validation = DataValidation(data_ingestion_artifact=dataingestionartifact, data_validation_config=data_validation_config)
        logging.info("Initiate Data Validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)
    except Exception as e:
        raise SuperStoreSalesException(e,sys)