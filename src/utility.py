import uuid
from bs4 import BeautifulSoup
import os
import shutil
import requests
from webapp.logging_layer.logger.logger import Applogger
import wget
import yaml
import importlib
import json
import zipfile
import random


def get_free_proxies():
    url = "https://free-proxy-list.net/"
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    proxies = []
    for row in soup.find('table', attrs={'id': 'proxylisttable'}).find_all('tr')[1:]:
        tds = row.find_all('td')
        try:
            ip = tds[0].text.strip()
            port = tds[1].text.strip()
            host = f"{ip}:{port}"
            proxies.append(host)
        except IndexError:
            continue
    return proxies


def get_session():
    proxies = get_free_proxies()
    session = requests.Session()
    proxy = random.choice((proxies))
    session.proxies = {'http': proxy, 'https': proxy}
    return session


def read_params(config_path: str) -> dict:
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def class_for_name(module_name, class_name):
    try:
        # load the module, will raise ImportError if module cannot be loaded
        module = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        class_ref = getattr(module, class_name)
        return class_ref
    except Exception as e:
        raise e


def values_from_schema_function(schema_path):
    try:
        with open(schema_path, 'r') as r:
            schema = json.load(r)
            r.close()

        pattern = schema['SampleFileName']
        length_of_date_stamp_in_file = schema['LengthOfDateStampInFile']
        length_of_time_stamp_in_file = schema['LengthOfTimeStampInFile']
        column_names = schema['ColName']
        number_of_columns = schema['NumberofColumns']
        return pattern, length_of_date_stamp_in_file, length_of_time_stamp_in_file,\
            column_names, number_of_columns

    except ValueError:
        raise ValueError
    except KeyError:
        raise KeyError
    except Exception as e:
        raise e


def create_directory_path(path, recreate=True):
    """
    :param path:
    :param recreate: Default it will delete the existing directory yet you can pass
    it's value to false if you do not want to remove existing directory
    :return:
    """

    try:
        if recreate:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=False)
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        raise e


def clean_data_source_dir(path, logger=None, enable_logging=True):
    """
    clean source directory.
    """
    try:
        logger.enable_logging = enable_logging
        if not os.path.exists(path):
            os.mkdir(path)
        for files in os.listdir(path):
            if '.gitignore' in files:
                pass
            logger.log(f"{os.path.join(path,files)} file will be deleated.")
            os.remove(os.path.join(path, files))
            logger.log(f"{os.path.join(path,files)} file has been deleated.")
    except Exception as e:
        raise e


def get_data_links(url, file_type='.zip'):
    """
    get download links
    """
    file_links = []
    soup = BeautifulSoup(requests.get(url).content, features='xml')
    for link in soup.find_all("Key"):
        filename = link.getText()
        if filename.endswith(file_type):
            file_links.append(url+filename)
    return file_links


def download_file_from_url(download_url, data_directory_path):
    try:
        filename = download_url.split('/')[-1][:-4] + '.csv'
        if not os.path.exists(os.path.join(data_directory_path, filename)):
            wget.download(download_url, out=data_directory_path)
    except Exception as e:
        raise e


def download_files_from_urls(page_url, destination_download_path,
                             no_of_files='all', logger=None, enable_logging=True):
    """
    download_files_from_url(): It will download file from url to your system
    ====================================================================================
    :param page_url: page url link
    :param local_system_directory_file_download_path: local system path where file
    has to be downloaded
    :param no_of_files: "all" to download all files, int(n) to download recent n-1 files
    last file will be used as prediction
    :return: bool
    """
    try:
        logger.enable_logging = enable_logging
        file_links = get_data_links(page_url)
        if no_of_files == 'all':
            logger.log(
                f"{no_of_files} file(s) will be downloaded from {page_url} to \
                {destination_download_path}.")
            for file_link in file_links:
                download_file_from_url(download_url=file_link,
                                       data_directory_path=destination_download_path)
            logger.log(
                f"{no_of_files} file(s) are downloaded from {page_url} to \
                {destination_download_path}.")
        else:
            for file_link in file_links[-no_of_files:]:
                logger.log(f"Most recent {no_of_files} file(s) will be downloaded \
                    from {page_url} to {destination_download_path}.")

                download_file_from_url(download_url=file_link,
                                       data_directory_path=destination_download_path)

                logger.log(f"Most recent {no_of_files} file(s) are downloaded \
                           from {page_url} to {destination_download_path}.")
            return True
        return False
    except Exception as e:
        raise e


def extract_zip_to_csv_file(source_file_path, destination_path):
    try:
        with zipfile.ZipFile(source_file_path, 'r') as zip_ref:
            listOfFileNames = zip_ref.namelist()
            for fileName in listOfFileNames:
                if fileName.endswith('.csv'):
                    zip_ref.extract(fileName, destination_path)
    except Exception as e:
        raise e


def extract_all_zip_to_csv_file(source_files_path, destination_path,
                                logger, enable_logging=True):
    """
    exracts all .csv file in a directory and remove the zip files
    """
    try:
        logger.enable_logging = enable_logging
        for item in os.listdir(source_files_path):
            if item.endswith('.zip'):
                file_path = os.path.abspath(
                    os.path.join(source_files_path, item))
                extract_zip_to_csv_file(file_path, destination_path)
                logger.log(f" File {item} is extracted from {source_files_path} to \
                    {destination_path}.")
                os.remove(file_path)
                logger.log(
                    f"Done! File {item} is removed from {source_files_path}.")
    except Exception as e:
        raise e


"""
def download_file_from_cloud(cloud_provider, cloud_directory_path,
                             local_system_directory_file_download_path,
                             logger,
                             is_logging_enable=True):
    '''
    download_training_file_from_s3_bucket(): It will download file from cloud storage to your system
    ====================================================================================================================
    :param cloud_provider: name of cloud provider amazon,google,microsoft
    :param cloud_directory_path: path of file located at cloud don't include bucket name
    :param local_system_directory_file_download_path: local system path where file has to be downloaded
    ====================================================================================================================
    :return: True if file downloaded else False
    '''
    try:

        logger.is_log_enable = is_logging_enable
        file_manager = FileManager(cloud_provider=cloud_provider)
        response = file_manager.list_files(directory_full_path=cloud_directory_path)
        if not response['status']:
            return True
        is_files_downloaded = 1
        for file_name in response['files_list']:
            logger.log(f"{file_name}file will be downloaded in dir--> {local_system_directory_file_download_path}.")
            response = file_manager.download_file(directory_full_path=cloud_directory_path,
                                                  local_system_directory=local_system_directory_file_download_path,
                                                  file_name=file_name)
            is_files_downloaded = is_files_downloaded * int(response['status'])
            logger.log(f"{file_name}file has been downloaded in dir--> {local_system_directory_file_download_path}.")
        return bool(is_files_downloaded)
    except Exception as e:
        raise e
"""


def get_logger_object_of_training(config_path, collection_name,
                                  execution_id=None, executed_by=None):
    config = read_params(config_path)
    database_name = config['log_database']['training_database_name']
    project_id = int(config['base']['project_id'])
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    if executed_by is None:
        executed_by = "Abhishek Jana"
    logger = Applogger(project_id=project_id, log_database=database_name,
                       log_collection_name=collection_name, execution_id=execution_id,
                       executed_by=executed_by)
    return logger


def get_logger_object_of_prediction(config_path, collection_name,
                                    execution_id=None, executed_by=None):
    config = read_params(config_path)
    database_name = config['log_database']['training_database_name']
    project_id = int(config['base']['project_id'])
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    if executed_by is None:
        executed_by = "Abhishek Jana"
    logger = Applogger(project_id=project_id, log_database=database_name,
                       log_collection_name=collection_name, execution_id=execution_id,
                       executed_by=executed_by)
    return logger
