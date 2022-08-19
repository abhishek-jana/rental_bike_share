import os
import sys
import argparse

from src.utility import read_params, create_directory_path
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from src.utility import get_logger_object_of_training
from webapp.exception_layer.generic_exception.generic_exception import GenericException

LOG_COLLECTION_NAME = "data_export"


class DataExporter:
    def __init__(self, config, logger, enable_logging):
        try:
            self.config = config
            self.logger = logger
            self.enable_logging = enable_logging
            self.mongo_db = MongoDBOperation()
            self.dataset_database = self.config["dataset"]["database_detail"]["training_database_name"]
            self.dataset_daily_collection_name = self.config["dataset"][
                "database_detail"]["dataset_daily_training_collection_name"]
            self.dataset_hourly_collection_name = self.config["dataset"][
                "database_detail"]["dataset_hourly_training_collection_name"]
            self.training_file_from_db = self.config["artifacts"]['training_data']['training_file_from_db']
            self.input_csv_day = self.config["artifacts"]['training_data']['input_csv_day']
            self.input_csv_hour = self.config["artifacts"]['training_data']['input_csv_hour']
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataExporter.__name__,
                        self.__init__.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def export_dataframe_from_database(self):
        try:
            create_directory_path(self.training_file_from_db)
            self.logger.log(f"Creating day dataframe of data stored in db"
                            f"[{self.dataset_database}] and collection[{self.dataset_daily_collection_name}]")
            day_df = self.mongo_db.get_dataframe_of_collection(db_name=self.dataset_database,
                                                               collection_name=self.dataset_daily_collection_name)
            input_csv_day_file_path = os.path.join(
                self.training_file_from_db, self.input_csv_day)
            self.logger.log(f"input csv day file will be generated at "
                            f"{input_csv_day_file_path}.")
            day_df.to_csv(input_csv_day_file_path, index=None, header=True)

            self.logger.log(f"Creating hour dataframe of data stored in db"
                            f"[{self.dataset_database}] and collection[{self.dataset_hourly_collection_name}]")
            hour_df = self.mongo_db.get_dataframe_of_collection(db_name=self.dataset_database,
                                                                collection_name=self.dataset_hourly_collection_name)
            input_csv_hour_file_path = os.path.join(
                self.training_file_from_db, self.input_csv_hour)
            self.logger.log(f"input csv hour file will be generated at "
                            f"{input_csv_hour_file_path}.")
            hour_df.to_csv(input_csv_hour_file_path, index=None, header=True)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataExporter.__name__,
                        self.export_dataframe_from_database.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e


def export_main(config_path, enable_logging=True, execution_id=None,
                executed_by=None):
    try:
        logger = get_logger_object_of_training(config_path=config_path,
                                               collection_name=LOG_COLLECTION_NAME,
                                               execution_id=execution_id,
                                               executed_by=executed_by)
        logger.enable_logging = enable_logging
        config = read_params(config_path)
        data_exporter = DataExporter(config=config, logger=logger,
                                     enable_logging=enable_logging)
        logger.log("Generating csv file from dataset stored in database.")
        data_exporter.export_dataframe_from_database()
        logger.log("Dataset has been successfully exported in directory \
            and exiting export pipeline.")
    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
            .format(export_main.__module__,
                    export_main.__name__))
        raise Exception(
            generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", default=os.path.join("config", "params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    export_main(config_path=parsed_args.config)
