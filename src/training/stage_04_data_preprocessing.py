import sys
import os
import argparse
import joblib
import pandas as pd

# from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from webapp.exception_layer.generic_exception.generic_exception import GenericException
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from src.utility import get_logger_object_of_training
from src.utility import read_params, create_directory_path


LOG_COLLECTION_NAME = "data_preprocessor"


class DataPreProcessing:
    """
    data preprocessing class for day and hour df
    """

    def __init__(self, logger, enable_logging=True):
        try:
            self.logger = logger
            self.logger.enable_logging = enable_logging
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.__init__.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def remove_columns(self, data, columns):
        """
        remove colums from pandas df

        Args:
            data (df): pandas df
            columns (list): list of columns to remove

        Raises:
            Exception: generic error

        Returns:
            df: pandas df
        """
        try:
            useful_data = data.drop(labels=columns, axis=1)
            return useful_data
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.remove_columns.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def scale_data(self, data, path, scaler_file_name, columns, is_dataframe_format_required=False, is_new_scaling=True):
        """
        data: dataframe to perform scaling
        path: path to save scaler object
        get_dataframe_format: default scaled output will be return as ndarray but if is true you will get
        dataframe format
        is_new_scaling: default it will create new scaling object and perform transformation.
        if it is false it will load scaler object from mentioned path paramter
        """
        self.logger.log(
            f'Cteating standard scaling on {columns} columns.')
        try:
            path = os.path.join(path)
            if not is_new_scaling:
                if os.path.exists(path):
                    scaler = joblib.load(os.path.join(path, scaler_file_name))
                    output = scaler.transform(data[columns])
                else:
                    raise Exception(
                        f"Scaler object is not found at path: {path}")
            else:
                scaler = StandardScaler()
                output = scaler.fit_transform(data[columns])
                create_directory_path(path, recreate=False)
                joblib.dump(scaler, os.path.join(path, scaler_file_name))
            if is_dataframe_format_required:
                output = pd.DataFrame(output, columns=columns)
            self.logger.log(
                f'Finished cteating standard scaling on {columns} columns.')
            return output
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.scale_data.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def one_hot_encoder(self, df, columns):
        """
        create one hot encoder

        Args:
            df (df): pandas df
            columns (list): list of categorical columns

        Raises:
            Exception: generic error

        Returns:
            df: one hot encoded df
        """
        self.logger.log(
            f'Cteating one hot encoding on {columns} columns.')
        try:
            cat_data = pd.get_dummies(
                df[columns], columns=columns, drop_first=True)
            self.logger.log(
                f'Finished one hot encoding on {columns} columns.')
            return cat_data
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.one_hot_encoder.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def join_dataframe(self, df1, df2):
        """
        join two pandas df

        Args:
            df1 (df): pandas df
            df2 (df): pandas df

        Raises:
            Exception: generic error

        Returns:
            df: joined df
        """
        try:
            df1.reset_index(drop=True, inplace=True)
            df2.reset_index(drop=True, inplace=True)
            return pd.concat([df1, df2], axis=1)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.join_dataframe.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def is_null_present(self, data, null_value_path, filename):
        """
        checks for null value and saves a df in a folder containing null value info

        Args:
            data (df): pandas df
            null_value_path (str): path to save df

        Raises:
            Exception: generic error

        Returns:
            bool: True of False
        """
        null_present = False
        self.logger.log(
            'Entered the is_null_present method of the Preprocessor class')
        try:
            null_counts = data.isna().sum()
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if null_present:  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing_values_count'] = np.asarray(
                    null_counts)
                create_directory_path(null_value_path, recreate=False)
                dataframe_with_null.to_csv(os.path.join(
                    null_value_path, filename))
                # storing the null column information to file
            self.logger.log("Finding missing values is a success.Data written to the null values file. Exited the "
                            "is_null_present method of the Preprocessor class")
            return null_present
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.is_null_present.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def impute_missing_values_KNN(self, df, columns):
        """
        impute missing values using KNN imputer

        Args:
            df (df): pandas df
            columns (str): list of columns

        Raises:
            Exception: generic error

        Returns:
            df: pandas df
        """
        self.logger.log(
            f'Started KNN imputation on :{columns} column(s).')
        try:
            imputer = KNNImputer()
            # columns = ['minTemp','maxTemp','precipitation']
            df[columns] = imputer.fit_transform(df[columns])
            self.logger.log(
                f'Successfully finished KNN imputation on :{columns} column(s).')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.impute_missing_values_KNN.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def impute_missing_values_mean(self, df, columns):
        """
        impute missing values using mean imputer

        Args:
            df (df): pandas df
            columns (str): list of columns

        Raises:
            Exception: generic error

        Returns:
            df: pandas df
        """
        self.logger.log(
            f'Started mean imputation on :{columns} column(s).')
        try:
            # columns = ['pressure']
            imputer = SimpleImputer(strategy='mean')
            df[columns] = imputer.fit_transform(df[columns])
            self.logger.log(
                f'Successfully finished mean imputation on :{columns} column(s).')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.impute_missing_values_mean.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def impute_missing_values_median(self, df, columns):
        """
        impute missing values using median imputer

        Args:
            df (df): pandas df
            columns (str): list of columns

        Raises:
            Exception: generic error

        Returns:
            df: pandas df
        """
        self.logger.log(
            f'Started meadian imputation on :{columns} column(s).')
        try:
            # columns = ['maxSteadyWind']
            imputer = SimpleImputer(strategy='median')
            df[columns] = imputer.fit_transform(df[columns])
            self.logger.log(
                f'Successfully finished median imputation on :{columns} column(s).')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.impute_missing_values_median.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def impute_missing_values_mode(self, df, columns):
        """
        impute missing values using mode imputer

        Args:
            df (df): pandas df
            columns (str): list of columns

        Raises:
            Exception: generic error

        Returns:
            df: pandas df
        """
        self.logger.log(
            f'Started mode imputation on :{columns} column(s).')
        try:
            # columns = ['description']
            imputer = SimpleImputer(strategy='most_frequent')
            df[columns] = imputer.fit_transform(df[columns])
            self.logger.log(
                f'Successfully finished mode imputation on :{columns} column(s).')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.impute_missing_values_mode.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def log_transform(self, df, columns):
        """
        log transformation on columns

        Args:
            df (df): pandas df
            columns (list): list of columns
        """
        self.logger.log(
            f'Started log transformation on :{columns} column(s).')
        try:
            df[columns] = np.log(df[columns])
            self.logger.log(
                f'Successfully finished log transformation on :{columns} column(s).')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.log_transform.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def create_windgust_info(self, df):
        """
        split windgust column to info given and not given

        Args:
            df (df): pandas df

        Raises:
            Exception: generic error

        Returns:
            df: pandas df with new column -windgust_info-
        """
        self.logger.log(
            f'Started splitting -maxWindGust- column into info given or not.')
        try:
            condition = [~df["maxWindGust"].isna(), df["maxWindGust"].isna()]
            value = ['yes', 'no']
            df['windgust_info'] = np.select(condition, value)
            df = df.drop(columns=['maxWindGust'], axis=1)
            self.logger.log(
                f'Finished splitting -maxWindGust- column. New column -windgust_info- created.')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.create_windgust_info.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def create_precipitation_info(self, df):
        """
        split precipitation column to no, medium and severe

        Args:
            df (df): pandas df

        Raises:
            Exception: generic error

        Returns:
            df: pandas df with new column -precipitation_info-
        """
        self.logger.log(
            f'Started splitting -precipitation- column into categories.')
        try:
            condition = [df["precipitation"] == 0,
                         (df["precipitation"] > 0) &
                         (df["precipitation"] < 5),
                         df["precipitation"] > 5]
            value = ['no', 'medium', 'severe']
            df['precipitation_info'] = np.select(condition, value)
            self.logger.log(
                f'Finished splitting -precipitaion- column. New column -precipitation_info- created.')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.create_precipitation_info.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def impute_wind_dir_deg(self, df):
        """
        custom imputation on wind_dir_degree from hour df

        Args:
            df (df): pandas df

        Raises:
            Exception: generic error

        Returns:
            df: pandas df after imputation with category value closest to nan
        """
        try:
            test_df = abs(df.groupby(['wind_dir_deg'])['member_casual'].mean() -
                          df['member_casual'][df["wind_dir_deg"].isna()].mean()).reset_index()
            value = test_df['wind_dir_deg'][test_df.member_casual ==
                                            test_df.member_casual.min()].values[0]
            df['wind_dir_deg'].fillna(value=value, inplace=True)
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.impute_wind_dir_deg.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def drop_columns_with_all_missing_value(self, df, threshold=0.8):
        """
        drop column with more than a threshold value missing

        Args:
            df (df): pandas df
            threshold (float, optional): threshhold value. Defaults to 0.8.

        Raises:
            Exception: generic error

        Returns:
            df: pandas df with missing value column dropped
        """
        self.logger.log(
            f'dropping columns with more than {threshold} missing values.')
        try:
            columns = df.columns
            for column in columns:
                if (df[column].isnull().sum()) / len(df[column]) > threshold:
                    df = df.drop(columns=column, axis=1)
                self.logger.log(
                    f'Dropping column {column}.')
            self.logger.log(
                f'Finished dropping column.')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.drop_columns_with_all_missing_value.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def merge_categories(self, df, new_cat_name, threshold_percent=5, column='description'):
        """
        merge categories with very simmiar and less contibution

        Args:
            df (df): pandas df
            new_cat_name (str): category name
            threshold_percent (int, optional): threshold value. Defaults to 5.
            column (str, optional): column of the df. Defaults to 'description'.

        Raises:
            Exception: generic error

        Returns:
            df: pandas df
        """
        self.logger.log(
            f'Merging descrition columns with less than  percent contribution')
        try:
            # new_cat_name = 'severe'
            series = pd.value_counts(df[column])
            mask = (series / series.sum() * 100).lt(threshold_percent)
            df = df.assign(weather_condition=np.where(
                df[column].isin(series[mask].index), new_cat_name, df[column]))
            df.drop(columns=[column], axis=1, inplace=True)
            self.logger.log(
                f'Finished merging descrition columns')
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataPreProcessing.__name__,
                        self.merge_categories.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e


class ModelTrainer:
    def __init__(self, config, logger, enable_logging=True):
        try:
            self.config = config
            self.logger = logger
            self.logger.enable_logging = enable_logging
            self.mongo_db = MongoDBOperation()
            self.training_file_path = self.config["artifacts"]['training_data']['training_file_from_db']
            self.input_csv_day = self.config["artifacts"]['training_data']['input_csv_day']
            self.input_csv_hour = self.config["artifacts"]['training_data']['input_csv_hour']
            self.target_columns = self.config['target_columns']['columns']
            self.null_value_file_path = self.config["artifacts"]["training_data"]["null_value_info_file_path"]
            self.scaler_path = self.config['artifacts']['training_data']['scaler_path']

            self.categorical_cols_hour = self.config["dataset"]["hour"]["categorical"]
            self.numerical_cols_hour = self.config["dataset"]["hour"]["numerical"]
            self.log_transform_hour = self.config["dataset"]["hour"]["log_transform"]
            self.categorical_cols_day = self.config["dataset"]["day"]["categorical"]
            self.numerical_cols_day = self.config["dataset"]["day"]["numerical"]
            self.log_transform_day = self.config["dataset"]["day"]["log_transform"]

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, ModelTrainer.__name__,
                        self.__init__.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def get_dataframe(self):
        try:
            day_file_path = os.path.join(
                self.training_file_path, self.input_csv_day)
            hour_file_path = os.path.join(
                self.training_file_path, self.input_csv_hour)
            day_df = pd.read_csv(day_file_path)
            hour_df = pd.read_csv(hour_file_path)
            return hour_df, day_df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, ModelTrainer.__name__,
                        self.get_dataframe.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def data_preperation(self):
        try:
            data_frame_hour, data_frame_day = self.get_dataframe()
            preprocess = DataPreProcessing(logger=self.logger,
                                           enable_logging=self.logger.enable_logging)

            input_features_day, target_features_day = data_frame_day.drop(
                self.target_columns, axis=1), data_frame_day[self.target_columns]

            # hour data
            is_null_present_hour = preprocess.is_null_present(
                data_frame_hour, self.null_value_file_path, "null_values_hour.csv")
            if is_null_present_hour:
                hour_cols = ['temperature', 'rel_temperature', 'rel_humidity',
                             'dew_point', 'pressure', 'icon', 'description']
                data_frame_hour = data_frame_hour.dropna(
                    subset=hour_cols)
                data_frame_hour = preprocess.impute_missing_values_median(
                    data_frame_hour, ['wind_speed'])
                data_frame_hour = preprocess.log_transform(
                    df=data_frame_hour, columns=self.log_transform_hour)
                data_frame_hour = preprocess.impute_wind_dir_deg(
                    data_frame_hour)

            input_features_hour, target_features_hour = data_frame_hour.drop(
                self.target_columns, axis=1), data_frame_hour[self.target_columns]
            input_features_hour = preprocess.drop_columns_with_all_missing_value(
                input_features_hour)
            input_features_hour = preprocess.remove_columns(
                input_features_hour, ['date', 'year', 'description'])

            input_features_hour_numerical = preprocess.scale_data(data=input_features_hour, path=self.scaler_path,
                                                                  scaler_file_name='scaler_hour.sav',
                                                                  columns=self.numerical_cols_hour,
                                                                  is_dataframe_format_required=True)
            input_features_hour_categorical = preprocess.one_hot_encoder(df=input_features_hour,
                                                                         columns=self.categorical_cols_hour)
            input_features_hour = preprocess.join_dataframe(input_features_hour_categorical,
                                                            input_features_hour_numerical)
            # day data
            is_null_present_day = preprocess.is_null_present(
                input_features_day, self.null_value_file_path, "null_values_day.csv")
            if is_null_present_day:
                input_features_day = preprocess.impute_missing_values_KNN(
                    input_features_day, ['minTemp', 'maxTemp', 'precipitation'])
                input_features_day = preprocess.impute_missing_values_mean(
                    input_features_day, ['pressure'])
                input_features_day = preprocess.impute_missing_values_median(
                    input_features_day, ['maxSteadyWind'])
                input_features_day = preprocess.impute_missing_values_mode(
                    input_features_day, ['description'])
            input_features_day = preprocess.drop_columns_with_all_missing_value(
                input_features_day)
            input_features_day = preprocess.merge_categories(
                input_features_day, new_cat_name='severe')
            input_features_day = preprocess.create_windgust_info(
                input_features_day)
            input_features_day = preprocess.create_precipitation_info(
                input_features_day)
            input_features_day = preprocess.remove_columns(
                input_features_day, ['date', 'year'])
            input_features_day = preprocess.log_transform(
                df=input_features_day, columns=self.log_transform_day)
            input_features_day_numerical = preprocess.scale_data(data=input_features_day, path=self.scaler_path,
                                                                 scaler_file_name='scaler_day.sav',
                                                                 columns=self.numerical_cols_day,
                                                                 is_dataframe_format_required=True)
            input_features_day_categorical = preprocess.one_hot_encoder(df=input_features_day,
                                                                        columns=self.categorical_cols_day)
            input_features_day = preprocess.join_dataframe(input_features_day_categorical,
                                                           input_features_day_numerical)
            return input_features_hour, target_features_hour, input_features_day, target_features_day
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, ModelTrainer.__name__,
                        self.data_preperation.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e


def preprocess_main(config_path, enable_logging=True, execution_id=None, executed_by=None):
    try:
        logger = get_logger_object_of_training(config_path=config_path,
                                               collection_name=LOG_COLLECTION_NAME,
                                               execution_id=execution_id,
                                               executed_by=executed_by)
        logger.enable_logging = enable_logging

        config = read_params(config_path)
        logger.log('Start of Data Preprocessing on input_day.csv file')
        data_preprocessor = ModelTrainer(config=config, logger=logger,
                                         enable_logging=enable_logging)
        X_hour, y_hour, X_day, y_day = data_preprocessor.data_preperation()
        print(X_hour.head(), y_hour.head())
        print(X_hour.isna().sum())
        print(X_day.head())
    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
            .format(preprocess_main.__module__,
                    preprocess_main.__name__))
        raise Exception(
            generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", default=os.path.join("config", "params.yaml"))
    parsed_args = args.parse_args()
    print(parsed_args.config)
    preprocess_main(config_path=parsed_args.config)
