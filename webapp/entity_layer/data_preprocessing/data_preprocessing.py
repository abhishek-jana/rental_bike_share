import pandas as pd
import sys
import os
from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from src.utility import create_directory_path
from webapp.exception_layer.generic_exception.generic_exception import GenericException


class DataPreProcessing:
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

    def is_null_present(self, data, null_value_path):
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
                create_directory_path(null_value_path)
                dataframe_with_null.to_csv(os.path.join(
                    null_value_path, "null_values.csv"))
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

    def fill_missing_values_mean(self, df, columns):
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
                        self.fill_missing_values_mean.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def fill_missing_values_median(self, df, columns):
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
                        self.fill_missing_values_median.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def fill_missing_values_mode(self, df, columns):
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
                        self.fill_missing_values_mode.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def split_windgust_info(self, df):
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
                        self.split_windgust_info.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def split_precipitation_info(self, df):
        self.logger.log(
            f'Started splitting -precipitation- column into categories.')
        try:
            condition = [df["precipitation"] == 0,
                         (df["precipitation"] > 0) & (
                df["precipitation"] < 5),
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
                        self.split_precipitation_info.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def drop_columns_with_all_missing_value(self, df, threshold=0.8):
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
