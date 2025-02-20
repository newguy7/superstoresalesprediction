from superstoresales.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from superstoresales.entity.config_entity import DataValidationConfig
from superstoresales.exception.exception import SuperStoreSalesException
from superstoresales.logging.logger import logging
from superstoresales.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp, chi2_contingency

import pandas as pd
import numpy as np
import os
import sys

from superstoresales.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise SuperStoreSalesException(e,sys)

    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame) -> bool:
        """
        Validates whether the number of columns in the dataframe matches the schema configuration.

        Args:
            dataframe (pd.Dataframe): The dataframe to validate

        Returns:
            bool: True if the number of columns matches, False otherwise    
        """
        try:
            number_of_columns = len(self._schema_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def validate_numerical_columns(self,dataframe: pd.DataFrame) -> bool:
        """
        Validates if the Dataframe contains the required numerical columns as per the schema.

        Args:
            dataframe (pd.Dataframe): The dataframe to validate

        Returns:
            bool: True if all required numerical columns exist, False otherwise
        """
        try:
            # Get expected numerical columns from schema
            expected_numerical_columns = self._schema_config.get("numerical_columns", [])
            actual_numerical_columns = dataframe.select_dtypes(include=['number']).columns

            logging.info(f"Expected numerical columns: {expected_numerical_columns}")
            logging.info(f"Actual numerical columns: {list(actual_numerical_columns)}")

            # check if all expected numerical columns exist
            missing_columns = set(expected_numerical_columns) - set(actual_numerical_columns)
            if missing_columns == 0:
                return True
            return False        

        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def validate_categorical_columns(self,dataframe: pd.DataFrame) -> bool:
        """
        Validates if the Dataframe contains the required categorical columns as per the schema.

        Args:
            dataframe (pd.Dataframe): The dataframe to validate

        Returns:
            bool: True if all required categorical columns exist, False otherwise
        """
        try:
            # Get expected categorical columns from schema
            expected_categorical_columns = self._schema_config.get("categorical_columns", [])
            actual_categorical_columns = dataframe.select_dtypes(include=['object']).columns

            logging.info(f"Expected categorical columns: {expected_categorical_columns}")
            logging.info(f"Actual categorical columns: {list(actual_categorical_columns)}")

            # check if all expected categorical columns exist
            missing_columns = set(expected_categorical_columns) - set(actual_categorical_columns)
            if missing_columns == 0:
                return True
            return False        

        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def detect_dataset_drift(self, base_df, current_df, threshold = 0.05) -> bool:
        """
        Detects dataset drift for numerical and categorical columns.

        Args:
            base_df (pd.DataFrame): Baseline dataset.
            current_df (pd.DataFrame): Current dataset to compare against the baseline.
            threshold (float): Significance level for drift detection. Default is 0.05.

        Returns:
            bool: True if drift is detected in any column, False otherwise.
        """
        try:
            # Validate column consistency
            if not all(base_df.columns == current_df.columns):
                raise ValueError("Columns in base_df and current_df do not match.")
            
            # Initialize drift status
            status = True
            report = {}

            # Split columns into numerical and categorical
            numerical_columns = base_df.select_dtypes(include=[np.number]).columns
            categorical_columns = base_df.select_dtypes(exclude=[np.number]).columns

            # Check drift for numerical columns using KS test
            for column in numerical_columns:
                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                if d1.empty or d2.empty:
                    raise ValueError("Train or Test dataset is empty! Please check your data ingestion process.")

                is_sample_dist = ks_2samp(d1,d2)
                p_value = is_sample_dist.pvalue
                is_drift_detected = p_value < threshold

                if is_drift_detected:
                    # Update drift status if any column shows drift
                    status = False

                report.update({column:{
                    "type": "numerical",
                    "p_value": float(p_value),
                    "drift_status": is_drift_detected
                }})

            # Check drift for categorical columns using Chi-Square test
            for column in categorical_columns:
                base_counts = base_df[column].value_counts(normalize=True)
                current_counts = current_df[column].value_counts(normalize=True)

                # Align categories between the two datasets
                combined_categories = base_counts.index.union(current_counts.index)
                base_counts = base_counts.reindex(combined_categories, fill_value=0)
                current_counts = current_counts.reindex(combined_categories, fill_value=0)

                contingency_table = pd.DataFrame({
                    "base": base_counts,
                    "current": current_counts
                }).T

                chi2_test = chi2_contingency(contingency_table)
                p_value = chi2_test[1]
                is_drift_detected = p_value < threshold

                if is_drift_detected:
                    status = False  # Update drift status if any column shows drift

                report.update({column:{
                    "type": "categorical",
                    "p_value": float(p_value),
                    "drift_status": is_drift_detected
                }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return status

        except Exception as e:
            raise SuperStoreSalesException(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the data from train and test
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all columns.\n"

            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all columns.\n"

            # Validate if numerical columns exists
            status = self.validate_numerical_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all numerical columns.\n"

            status = self.validate_numerical_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all numerical columns.\n"

            # Validate if categorical columns exists
            status = self.validate_categorical_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"Train dataframe does not contain all categorical columns.\n"

            status = self.validate_categorical_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"Test dataframe does not contain all categorical columns.\n"

            # Check datadrift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return data_validation_artifact
        
        except Exception as e:
            raise SuperStoreSalesException(e,sys)