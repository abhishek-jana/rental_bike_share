from webapp.exception_layer.generic_exception.generic_exception import GenericException
from src.utility import read_params, create_directory_path, values_from_schema_function, get_logger_object_of_training
from webapp.project_library_layer.datetime_library.date_time import get_date, get_time
import os
import re
import shutil
import sys
import pandas as pd
import argparse

LOG_COLLECTION_NAME = "data_validator"


class DataValidator:
    """
    Data validator class to validate the data downloaded from source.
    """

    def __init__(self, config, logger, enable_logging=True):
        try:
            self.config = config
            self.logger = logger
            self.logger.enable_logging = enable_logging
            self.file_path = self.config["data_source"]["Training_Batch_Files"]
            self.good_file_path = self.config["artifacts"]["training_data"]["good_file_path"]
            self.bad_file_path = self.config["artifacts"]["training_data"]["bad_file_path"]
            self.archive_bad_file_path = self.config["artifacts"]["training_data"]["archive_bad_file_path"]
            self.training_schema_file = self.config["config"]["schema_training"]
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__, self.__init__.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def archive_bad_files(self):
        """
        create bad_file_eventdate_eventtime folder in archive directory

        Raises:
            Exception: generic error message
        """
        try:
            folder_name = f"bad_files_{get_date().replace('-','_')}_{get_time().replace(':','_')}"
            archive_directory_path = os.path.join(
                self.archive_bad_file_path, folder_name)
            create_directory_path(archive_directory_path)
            for files in os.listdir(self.bad_file_path):
                source_file_path = os.path.join(self.bad_file_path, files)
                shutil.move(source_file_path, archive_directory_path)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__,
                        self.archive_bad_files.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def create_good_bad_archive_bad_file_path(self):
        """
        create good, bad, arcvhive folders

        Raises:
            Exception: generic error message
        """
        try:
            create_directory_path(self.good_file_path)
            create_directory_path(self.bad_file_path)
            create_directory_path(self.archive_bad_file_path, recreate=False)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(),
                        DataValidator.__name__, self.create_good_bad_archive_bad_file_path.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def value_from_schema(self):
        """
        get the schema values from file

        Raises:
            Exception: generic error message
        """
        try:
            return values_from_schema_function(self.training_schema_file)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__, self.value_from_schema.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def file_name_regular_expression(self):
        """
        return filename in regular expression format

        Raises:
            Exception: generic error message
        """
        try:
            return r"[\d]+['-capitalbikeshare-tripdata.csv']"
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__, self.file_name_regular_expression.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def validate_file_name(self):
        """
        validate filename according to the schema

        Raises:
            Exception: generic error message
        """
        try:
            self.create_good_bad_archive_bad_file_path()
            regx_pattern = self.file_name_regular_expression()
            pattern, length_of_date_stamp_in_file, length_of_time_stamp_in_file,\
                column_names, number_of_columns = self.value_from_schema()
            self.logger.log("validating filenames")
            files = os.listdir(self.file_path)
            for filename in files:
                file_path = os.path.join(self.file_path, filename)
                split_at_dot = re.split(
                    '-capitalbikeshare-tripdata.csv', filename)
                if re.match(regx_pattern, filename) and \
                        len(split_at_dot[0]) == length_of_date_stamp_in_file:
                    destination_file_path = os.path.join(
                        self.good_file_path, filename)
                    self.logger.log(
                        f"file name : {filename} matched hence moving file\
                            to good file path {destination_file_path}")
                    shutil.move(file_path, destination_file_path)
                else:
                    destination_file_path = os.path.join(
                        self.bad_file_path, filename)
                    self.logger.log(
                        f"file name : {filename} did not match hence moving file\
                            to bad file path {destination_file_path}")
                    shutil.move(file_path, destination_file_path)

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__, self.validate_file_name.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def validate_missing_values_in_whole_column(self):
        """
        Check if an etire column has missing values, move the fie to bad folder if True

        Raises:
            Exception: Generic error message
        """
        try:
            self.logger.log("Missing Values Validation Started!!")
            for filename in os.listdir(self.good_file_path):
                csv = pd.read_csv(os.path.join(self.good_file_path, filename))
                count = 0
                for column in csv:
                    if len(csv[column]) - csv[column].count() == len(csv[column]):
                        count += 1
                        shutil.move(os.path.join(
                            self.good_file_path, filename), self.bad_file_path)
                        self.logger.log(
                            "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % filename)
                        break
                    if count == 0:
                        continue
                        # print(csv[column])
                        #     csv.rename(columns={"Unnamed: 0": "Premium "}, inplace=True)
                        # csv.to_csv(os.path.join(self.good_file_path, file), index=None, header=True)

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__,
                        self.validate_missing_values_in_whole_column.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def validate_no_of_column(self, no_of_columns):
        """
        Validating the number of columns. If the number of columns
        matches with the config file, move to good data directory.
        Else move to bad data directory.

        Args:
            no_of_column (int): total number of columns given by the cilent.

        Raises:
            Exception: Generic exception error message.
        """
        try:
            self.logger.log(
                "Validating the number of columns in the input file.")
            files = os.listdir(self.good_file_path)
            for filename in files:
                filepath = os.path.join(self.good_file_path, filename)
                df = pd.read_csv(filepath)
                if df.shape[1] != no_of_columns:
                    destination_file_path = os.path.join(
                        self.bad_file_path, filename)
                    self.logger.log(f"File: {filename} has incorrect number of columns,\
                        expected: {df.shape[1]}. Has: {no_of_columns}.\
                         hence moving file to bad file path {destination_file_path}")
                    shutil.move(filepath, destination_file_path)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__.__str__(), DataValidator.__name__, self.validate_no_of_column.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e


def validation_main(config_path, enable_logging=True, execution_id=None, executed_by=None):
    try:
        logger = get_logger_object_of_training(config_path=config_path,
                                               collection_name=LOG_COLLECTION_NAME,
                                               execution_id=execution_id,
                                               executed_by=executed_by)
        logger.enable_logging = enable_logging
        logger.log(
            "Starting data validation.\nReading configuration file.")

        config = read_params(config_path)
        data_validator = DataValidator(
            config=config, logger=logger, enable_logging=enable_logging)
        pattern, length_of_date_stamp_in_file, length_of_time_stamp_in_file, column_names,\
            number_of_columns = data_validator.value_from_schema()
        data_validator.validate_file_name()
        data_validator.validate_no_of_column(no_of_columns=number_of_columns)
        data_validator.validate_missing_values_in_whole_column()
        data_validator.archive_bad_files()

    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
            .format(validation_main.__module__.__str__(), validation_main.__name__))
        raise Exception(
            generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", default=os.path.join("config", "params.yaml"))
    args.add_argument("--datasource", default=None)
    parsed_args = args.parse_args()
    print("started")
    validation_main(config_path=parsed_args.config)
