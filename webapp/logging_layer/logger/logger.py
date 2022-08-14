from datetime import datetime
from webapp.project_library_layer.datetime_library import date_time

from webapp.data_access_layer.mongo_db.mongo_db_atlas import MongoDBOperation
from webapp.project_library_layer.initializer.initializer import Initializer
from webapp.exception_layer.logger_exception.logger_exception import AppLoggerException
import uuid
import sys


class Applogger:
    def __init__(self, project_id=True, log_database=None, log_collection_name=None,
                 executed_by=None, execution_id=None, socket_io=None, enable_logging=True):
        self.project_id = project_id
        self.log_database = log_database
        self.log_collection_name = log_collection_name
        self.executed_by = executed_by
        self.execution_id = execution_id
        self.mongo_db_object = MongoDBOperation()
        self.socket_io = socket_io
        self.enable_logging = enable_logging

    def log(self, log_message):
        if not self.enable_logging:
            return True
        log_writer_id = str(uuid.uuid4())
        log_data = None
        try:
            if self.socket_io is not None:
                if self.log_database == Initializer().get_training_database_name():
                    self.socket_io.emit("started_training" + str(self.project_id),
                                        {
                                            'message': "<span style='color:red'>executed_by [{}]</span>"
                                                       "<span style='color:#008cba;'> exec_id {}:</span> "
                                                       "<span style='color:green;'>{}</span> {} "
                                                       ">{}".format(self.executed_by, self.execution_id,
                                                                    date_time.get_date(),
                                                                    date_time.get_time(), log_message)}, namespace="/training_model")
                if self.log_database == Initializer().get_prediction_database_name():
                    self.socket_io.emit("prediction_started" + str(self.project_id),
                                        {
                                            'message': "<span style='color:red'>executed_by [{}]</span>"
                                                       "<span style='color:#008cba;'> exec_id {}:</span> "
                                                       "<span style='color:green;'>{}</span> {} "
                                                       ">{}".format(self.executed_by, self.execution_id,
                                                                    date_time.get_date(),
                                                                    date_time.get_time(), log_message)}, namespace="/predicting_model")

                # file_object = None
            self.now = datetime.now()
            self.date = datetime.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")
            log_data = {
                'log_updated_date': date_time.get_date(),
                'log_update_time': date_time.get_time(),
                'execution_id': self.execution_id,
                'message': log_message,
                'executed_by': self.executed_by,
                'project_id': log_writer_id,
                'updated_date_time': datetime.now()
            }
            self.mongo_db_object.insert_record_in_collection(
                self.log_database, self.log_collection_name, log_data
            )
        except Exception as e:

            app_logger_exception = AppLoggerException(
                "Failed to lof data file in module [{0}] class [{1}] method [{2}] --> log detail [{3}]"
                .format(Applogger.__module__.__str__(), Applogger.__name__,
                        self.log.__name__, log_data)
            )
            message = Exception(
                app_logger_exception.error_message_detail(str(e), sys))
            print(message)
