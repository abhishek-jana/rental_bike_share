import pandas as pd
import argparse
import holidays
import requests
from bs4 import BeautifulSoup
import os
import random
import sys
from time import sleep
from src.utility import get_session, read_params, get_logger_object_of_training, get_session
from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.exception_layer.generic_exception.generic_exception import GenericException

LOG_COLLECTION_NAME = "data_transformer"

# Todo
# function for hour_df to merge weather data with hour data


class DataTransformer:
    """
    DataTransformer class for transforming dataset
    """

    def __init__(self, config, logger, enable_logging):
        try:
            self.config = config
            self.logger = logger
            self.logger.enable_logging = enable_logging
            self.good_file_path = self.config["artifacts"]["training_data"]["good_file_path"]
            self.unwanted_column_names = self.config["dataset"]["unwanted_column"]
            self.datetime_column = self.config["dataset"]["convert_to_datetime"][0]
            self.target_column = self.config["target_columns"]["columns"][0]
            self.training_schema_file = self.config['config']['schema_training']
            self.years = [int(year[:4])
                          for year in os.listdir(self.good_file_path)]
            self.min_year = min(self.years)
            self.max_year = max(self.years)
            self.mongo_db = MongoDBOperation()
            self.dataset_database = self.config["dataset"]["database_detail"]["training_database_name"]
            self.dataset_daily_collection_name =\
                self.config["dataset"]["database_detail"]["dataset_daily_training_collection_name"]
            self.dataset_hourly_collection_name =\
                self.config["dataset"]["database_detail"]["dataset_hourly_training_collection_name"]
            self.mongo_db.drop_collection(
                self.dataset_database, self.dataset_daily_collection_name)
            self.mongo_db.drop_collection(
                self.dataset_database, self.dataset_hourly_collection_name)
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__,
                        self.__init__.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def remove_unwanted_columns(self, df):
        """
        remove columns from data

        Args:
            df (DataFrame): pandas DataFrame

        Raises:
            Exception: generic excetion error

        Returns:
            DataFrame: return pandas DataFrame after removing the columns
        """
        try:
            # print(self.unwanted_column_names)
            column_to_remove = list(
                filter(lambda x: x in df.columns, self.unwanted_column_names))
            if len(column_to_remove) > 0:
                return df.drop(column_to_remove, axis=1)
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.remove_unwanted_columns.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def split_date_hour(self, df):
        """
        split the datetime column in date, hour and dayofweek columns and

        Args:
            df (DataFrame): pandas DataFrame

        Raises:
            Exception: generic error message is

        Returns:
            DataFrame: pandas dataframe with date and hour columns added
            and datetime column dropped
        """
        try:
            df[self.datetime_column] = pd.to_datetime(
                df[self.datetime_column], infer_datetime_format=True)
            # print(df[self.datetime_column])
            df["hour"] = df[self.datetime_column].dt.hour
            df["date"] = df[self.datetime_column].dt.date
            df["dayofweek"] = df[self.datetime_column].dt.isocalendar().day
            return df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.split_date_hour.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def get_daily_hourly_bike_rental_count(self, df):
        """
        Group data by daily and hourly on target columns

        Args:
            df (Dataframe): pandas DataFrame 

        Raises:
            Exception: generic error message    

        Returns:
            DataFrame: returns 2 pandas dataframe: hour_df and day_df
        """
        try:
            # df = self.split_date_hour(df)
            day_df = df.groupby(["date", "dayofweek"], as_index=False)[
                self.target_column].count()
            hour_df = df.groupby(["hour", "date", "dayofweek"], as_index=False)[
                self.target_column].count()
            hour_df = hour_df.sort_values(by=["date", "hour"])
            return hour_df, day_df

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__,
                        self.get_daily_hourly_bike_rental_count.__name__))
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def list_of_holiday(self):
        """
        returns the list of holidays in Washington D.C 

        Returns:
            list: list of holidays
        """
        try:
            # DC and US are hardcoded
            holiday_list = []
            us_dc_holidays = holidays.country_holidays(
                'US', years=[self.min_year, self.max_year], expand=True, observed=True, subdiv='DC')
            for holiday in us_dc_holidays.items():
                holiday_list.append(holiday[0])
            return holiday_list

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.list_of_holiday.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def is_holiday(self, row):
        """
        checks a row containing date and dayofweek values

        Args:
            row (dict): dict containing date and dayofweek values

        Raises:
            Exception: generic error message

        Returns:
            int: 1 if holiday 0 if not
        """
        try:
            holidaylist = self.list_of_holiday()
            result = bool(
                all([row["date"] in holidaylist, row['dayofweek'] != 7]))
            return int(result)

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.is_holiday.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def add_holiday_column(self, df):
        """
        adds holiday column to

        Args:
            df (DataFrame): pandas DataFrame

        Raises:
            Exception: gerenric error message

        Returns:
            DataFrame: pandas DataFrame
        """
        try:
            # hour_df, day_df=self.get_daily_hourly_bike_rental_count(df)
            df['holiday'] = df.apply(self.is_holiday, axis=1)
            return df

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.add_holiday_column.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def split_date_column(self, df_with_date_column):
        """
        splits date column and adds day, week, month, year columns to

        Args:
            df_with_date_column (DataFrame): pandas df DataFrame

        Raises:
            Exception: Generic error message

        Returns:
            df: pandas df with day,week,month, year columns dropping date column
        """
        try:
            df_with_date_column["date"] = pd.to_datetime(
                df_with_date_column["date"])
            df_with_date_column["day"] = df_with_date_column["date"].dt.day
            df_with_date_column["week"] = df_with_date_column["date"].dt.isocalendar(
            ).week
            df_with_date_column["month"] = df_with_date_column["date"].dt.month
            df_with_date_column["year"] = df_with_date_column["date"].dt.year
            # df_with_date_column.drop(columns=["date"], inplace=True)
            return df_with_date_column

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.split_date_column.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def add_season(self, df_with_day_month_column):
        """
        add season column: 1 -> winter, 2 -> spring, 3 -> summer, 4 -> autumn

        Args:
            df_with_day_month_column (df): pandas df DataFrame

        Raises:
            Exception: generic error message

        Returns:
            df: df with season column added
        """
        try:
            # 1: winter; 2: spring; 3: summer; 4: autumn
            season = df_with_day_month_column["month"] % 12 // 3 + 1
            season.loc[(df_with_day_month_column["month"].isin((3, 6, 9, 12))) &
                       (df_with_day_month_column["day"] < 21) & (df_with_day_month_column['week'] < 48)] -= 1

            df_with_day_month_column["season"] = season

            return df_with_day_month_column
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.add_season.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def get_daily_weather(self, month, year):
        """
        get daily weather report from a specific month of a year of

        Args:
            month (int): month
            year (int): year

        Raises:
            Exception: generic exception message

        Returns:
            df: pandas df with day, minTemp, maxTemp, maxSteadyWind, precipitaion, snowDepth,
            pressure, description columns
        """
        try:
            # url is hardcoded
            # s = requests.session()
            url = f"https://i-weather.com/weather/washington/history/monthly-history/?gid=4140963&station=19064&month={month}&year={year}&language=english&country=us-united-states"

            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                 AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"})
            soup = BeautifulSoup(r.text, 'lxml')
            weather = soup.find('table', class_="monthly-history")
            rows = weather.findAll('tr')

            day = []
            minTemp = []
            maxTemp = []
            maxSteadyWind = []
            maxWindGust = []
            precipitation = []
            snowDepth = []
            pressure = []
            # icon = []
            description = []

            for row in rows:
                cells = row.findAll("td")
                if len(cells) > 0:
                    day.append(pd.to_datetime(cells[0].text))
                    minTemp.append(cells[1].text)
                    maxTemp.append(cells[2].text)
                    maxSteadyWind.append(cells[3].text)
                    maxWindGust.append(cells[4].text)
                    precipitation.append(cells[5].text)
                    snowDepth.append(cells[6].text)
                    pressure.append(cells[7].text)
                    # icon.append(cells[8].span.attrs['data-icon'] if cells[8].span else None)
                    description.append(cells[9].text)

            data = {
                "date": day,
                "minTemp": minTemp,
                "maxTemp": maxTemp,
                "maxSteadyWind": maxSteadyWind,
                "maxWindGust": maxWindGust,
                "precipitation": precipitation,
                "snowDepth": snowDepth,
                "pressure": pressure,
                # "icon" : icon,
                "description": description
            }
            return pd.DataFrame(data, columns=data.keys())
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.get_daily_weather.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def get_hourly_weather(self, date):
        """
        get hourly weather report from a specific date

        Args:
            date (dt): datetime format eg. 2022-08-02

        Raises:
            Exception: generic exception message

        Returns:
            df: pandas df with "date","temperature","rel_tmperature",
            "wind","wind_gust","rel_humidity","dew_point","pressure",
            "icon","description" columns
        """
        try:
            # s = requests.session()

            url = f"https://i-weather.com/weather/washington/history/daily-history/?gid=4140963&date={date}&station=19064&language=english&country=us-united-states"

            r = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36"})
            soup = BeautifulSoup(r.text, 'lxml')

            # weather = soup.find('table', class_="daily-history")

            rows = soup.findAll('tr')

            hours = []
            dates = []
            temperature = []
            rel_tmperature = []
            wind = []
            wind_gust = []
            rel_humidity = []
            dew_point = []
            pressure = []
            icon = []
            description = []
            for row in rows:
                cells = row.findAll("td")
                if len(cells) == 10:
                    hour = int(cells[0].text[:2])
                    hours.append(hour)
                    dates.append(date)
                    temperature.append(cells[1].text)
                    rel_tmperature.append(cells[2].text)
                    wind.append(cells[3].text)
                    wind_gust.append(cells[4].text)
                    rel_humidity.append(cells[5].text)
                    dew_point.append(cells[6].text)
                    pressure.append(cells[7].text)
                    # icon = cells[8].span.attrs['data-icon']
                    icon.append(
                        cells[8].span.attrs['data-icon'] if cells[8].span else None)
                    description.append(
                        cells[9].find('span', "details").text)

            data = {
                "hour": hours,
                "date": dates,
                "temperature": temperature,
                "rel_temperature": rel_tmperature,
                "wind": wind,
                "wind_gust": wind_gust,
                "rel_humidity": rel_humidity,
                "dew_point": dew_point,
                "pressure": pressure,
                "icon": icon,
                "description": description
            }
            df = pd.DataFrame(data, columns=data.keys())
            df = df.drop_duplicates(subset=["hour"], keep='last')
            return df.sort_values(by=["hour"]).reset_index(drop=True)

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.get_hourly_weather.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def clean_daily_weather_data(self, input_df):
        """
        takes the weather df and cleans it into

        Args:
            input_df (df): pandas df

        Raises:
            Exception: generic error message

        Returns:
            df: pandas df
        """
        try:
            input_df["minTemp"] = input_df["minTemp"].str.replace('°C', '')
            input_df["maxTemp"] = input_df["maxTemp"].str.replace('°C', '')
            input_df["maxSteadyWind"] = input_df["maxSteadyWind"].str.replace(
                'Km/h', '')
            input_df["maxWindGust"] = input_df["maxWindGust"].str.replace(
                'Km/h', '')
            input_df["precipitation"] = input_df["precipitation"].str.replace(
                'mm', '')
            input_df["snowDepth"] = input_df["snowDepth"].str.replace('mm', '')
            input_df["pressure"] = input_df["pressure"].str.replace('mb', '')
            return input_df

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.clean_daily_weather_data.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def get_hourly_weathers(self, hour_df):
        """
        - Extract unique dates from the df
        - get hourly weather for each dates
        - merge the DataFrame 

        Args:
            hour_df (df): pandas DataFrame

        Raises:
            Exception: generic error message

        Returns:
            df: combined DataFrame
        """
        try:
            hourly_weather = []
            hour_df["date"] = pd.to_datetime(hour_df["date"]).dt.date
            dates = hour_df["date"].unique()
            for date in dates:
                hour_data = self.get_hourly_weather(date)
                sleep(random.random())
                hourly_weather.append(hour_data)
                hourly_weather_df = pd.concat(hourly_weather)
            return hourly_weather_df
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.get_hourly_weathers.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def clean_hourly_weather_data(self, input_df):
        """
        takes the weather df and cleans it into

        Args:
            input_df (df): pandas df

        Raises:
            Exception: generic error message

        Returns:
            df: pandas df
        """
        try:
            input_df["temperature"] = input_df["temperature"].str.replace(
                '°C', '')
            input_df["rel_temperature"] = input_df["rel_temperature"].str.replace(
                '°C', '')
            input_df["wind"] = input_df["wind"].str.replace(
                'Variable at ', 'NaN°')
            input_df["wind_gust"] = input_df["wind_gust"].str.replace(
                'Km/h', '')
            input_df[["wind_dir_deg", 'wind_speed']
                     ] = input_df['wind'].str.split("°", expand=True)
            input_df["wind_speed"] = input_df['wind_speed'].str.split(" ", expand=True)[
                0]
            input_df = input_df.drop(['wind'], axis=1)
            input_df["rel_humidity"] = input_df["rel_humidity"].str.replace(
                '%', '')
            input_df["dew_point"] = input_df["dew_point"].str.replace('°C', '')
            input_df["pressure"] = input_df["pressure"].str.replace('mb', '')
            return input_df

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.clean_hourly_weather_data.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def merge_day_and_weather_data(self, day_df, daily_weather_df):
        """
        merge the main data with weather data on day columns

        Args:
            day_df (df): pandas df
            daily_weather_df (df): pandas dfs

        Raises:
            Exception: generic error message

        Returns:
            df: pandas merged df
        """
        try:
            return day_df.merge(daily_weather_df, how='left', on='date')
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.merge_day_and_weather_data.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def merge_hour_and_weather_data(self, hour_df, hourly_weather_df):
        """
        merge the main data with weather data on day columns

        Args:
            hour_df (df): pandas df
            hourly_weather_df (df): pandas dfs

        Raises:
            Exception: generic error message

        Returns:
            df: pandas merged df
        """
        try:
            return hour_df.merge(hourly_weather_df, how='left', on=['hour', 'date'])
        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.merge_day_and_weather_data.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def preprocess_data(self, df):
        """
        preprocess the main data and fetch weather data and merge them on main data.

        Args:
            df (DataFrame): pandas DataFrame
            month (int): month from input file
            year (int): year from input file

        Raises:
            Exception: generic exception message

        Returns:
            df: merged Datframe.
        """
        try:
            df = self.split_date_hour(df)
            df = self.remove_unwanted_columns(df)
            hour_df, day_df = self.get_daily_hourly_bike_rental_count(df)
            hour_df = self.add_holiday_column(hour_df)
            hour_df = self.split_date_column(hour_df)
            hour_df = self.add_season(hour_df)

            day_df = self.add_holiday_column(day_df)
            day_df = self.split_date_column(day_df)
            day_df = self.add_season(day_df)

            return hour_df, day_df

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.preprocess_data.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e

    def unite_dataset(self):
        """
        - create day and hour dataframe
        - preprocess and merge the main data with cleaned weather data
        - insert the dataframes into mongodb collection

        Raises:
            Exception: generic exception
        """
        try:
            day_dataset_list = []
            day_weather_list = []

            hour_dataset_list = []
            hour_weather_list = []

            for file in os.listdir(self.good_file_path):
                month, year = file[4:6], file[:4]
                print(month, year)
                hour_df, day_df = self.preprocess_data(
                    df=pd.read_csv(os.path.join(self.good_file_path, file)))

                hour_dataset_list.append(hour_df)
                day_dataset_list.append(day_df)

                hour_weather_df = self.get_hourly_weathers(hour_df)
                hour_weather_df = self.clean_hourly_weather_data(
                    hour_weather_df)

                day_weather_df = self.get_daily_weather(month=month, year=year)
                day_weather_df = self.clean_daily_weather_data(day_weather_df)

                hour_weather_list.append(hour_weather_df)
                day_weather_list.append(day_weather_df)
                sleep(random.random())
                # print(hour_df)

            main_hour_df = pd.concat(hour_dataset_list)
            total_hour_weather_df = pd.concat(hour_weather_list)

            main_day_df = pd.concat(day_dataset_list)
            total_day_weather_df = pd.concat(day_weather_list)

            main_hour_df = self.merge_hour_and_weather_data(
                main_hour_df, total_hour_weather_df)
            main_day_df = self.merge_day_and_weather_data(
                main_day_df, total_day_weather_df)

            self.logger.log(f"Inserting dataset into database {self.dataset_database} "
                            f"collection_name: {self.dataset_hourly_collection_name}")
            self.mongo_db.insert_dataframe_into_collection(self.dataset_database,
                                                           self.dataset_hourly_collection_name, main_hour_df)

            self.logger.log(f"Inserting dataset into database {self.dataset_database} "
                            f"collection_name: {self.dataset_daily_collection_name}")
            self.mongo_db.insert_dataframe_into_collection(self.dataset_database,
                                                           self.dataset_daily_collection_name, main_day_df)

        except Exception as e:
            generic_exception = GenericException(
                "Error occurred in module [{0}] class [{1}] method [{2}]"
                .format(self.__module__, DataTransformer.__name__, self.unite_dataset.__name__)
            )
            raise Exception(
                generic_exception.error_message_detail(str(e), sys)) from e


def transform_main(config_path, enable_logging=True, execution_id=None,
                   executed_by=None):
    try:
        logger = get_logger_object_of_training(config_path=config_path, collection_name=LOG_COLLECTION_NAME,
                                               execution_id=execution_id, executed_by=executed_by)
        logger.enable_logging = enable_logging
        config = read_params(config_path)
        logger.log('Start of Data Preprocessing before DB')
        data_transformer = DataTransformer(
            config=config, logger=logger, enable_logging=enable_logging)
        # month,year = pattern[4:6],pattern[:4]
        data_transformer.unite_dataset()
        logger.log('Data Preprocessing before DB Completed !!')
    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
            .format(transform_main.__module__,
                    transform_main.__name__))
        raise Exception(
            generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", default=os.path.join("config", "params.yaml"))
    parsed_args = args.parse_args()
    print("started")
    transform_main(config_path=parsed_args.config)
