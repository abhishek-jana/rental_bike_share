import os
import sys
import argparse

from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from src.utility import get_logger_object_of_training

from webapp.entity_layer.data_preprocessing.data_preprocessing import DataPreProcessing

LOG_COLLECTION_NAME = "day_data_export"
