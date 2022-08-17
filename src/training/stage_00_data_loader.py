import os
import shutil
import sys
import argparse

from src.utility import get_logger_object_of_training, read_params
from src.utility import clean_data_source_dir, download_files_from_urls
from src.utility import extract_all_zip_to_csv_file
from webapp.exception_layer.generic_exception.generic_exception import GenericException


LOG_COLLECTION_NAME = "data_loader"


def loader_main(config_path, enable_logging=True, execution_id=None, executed_by=None):
    try:
        # creating logger object from Applogger
        logger = get_logger_object_of_training(config_path=config_path,
                                               collection_name=LOG_COLLECTION_NAME,
                                               execution_id=execution_id,
                                               executed_by=executed_by)
        logger.enable_logging = enable_logging
        logger.log(
            "Starting data loading operation.\nReading configuration file.")

        config = read_params(config_path)

        # cloud_provider = config['cloud_provider']['name']
        status = config['cloud_provider']['is_enabled']
        downloader_url = config['data_download']['core_data_url']
        downloader_path = config['data_download']['cloud_training_directory_path']
        download_path = config['data_source']['Training_Batch_Files']
        no_of_files = config["data_download"]["no_of_files"]
        logger.log(
            "Configuration detail has been fetched from configuration file.")

        logger.log(
            f"Cleaning local directory [{download_path}]  for training.")
        # removing existing file from local system
        clean_data_source_dir(download_path, logger=logger,
                              enable_logging=enable_logging)
        logger.log(
            f"Cleaning completed. Directory has been cleared now  [{download_path}]")
        # downloading training and additional training file from cloud into local system
        logger.log(
            f"Data will be downloaded from {downloader_url} to local system")
        if int(status):
            pass
            # download_file_from_cloud(cloud_provider=cloud_provider,
            #                          cloud_directory_path=downloader_path,
            #                          local_system_directory_file_download_path=download_path,
            #                          logger=logger,
            #                          enable_logging_enable=enable_logging
            #                          )
        else:
            download_files_from_urls(page_url=downloader_url,
                                     destination_download_path=downloader_path,
                                     no_of_files=no_of_files, logger=logger,
                                     enable_logging=enable_logging)
            logger.log("Download completed. Downloaded files will be extracted")
            extract_all_zip_to_csv_file(source_files_path=downloader_path,
                                        destination_path=downloader_path,
                                        logger=logger, enable_logging=True)
            logger.log("Extraction completed.")
            for filetype in os.listdir(downloader_path):
                if filetype.endswith('.csv'):
                    print(f"Source dir: {downloader_path} file: {filetype} is being copied into \
                        destination dir: {download_path}"
                          f" file: {filetype}")
                    shutil.copy(os.path.join(downloader_path, filetype),
                                os.path.join(download_path, filetype))
        logger.log(
            "Data has been downloaded from cloud storage into local system")

    except Exception as e:
        generic_exception = GenericException(
            "Error occurred in module [{0}] method [{1}]"
            .format(loader_main.__module__,
                    loader_main.__name__))
        raise Exception(
            generic_exception.error_message_detail(str(e), sys)) from e


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "--config", default=os.path.join("config", "params.yaml"))
    parsed_args = args.parse_args()
    print("strated")
    loader_main(config_path=parsed_args.config)
