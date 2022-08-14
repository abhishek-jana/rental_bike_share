import yaml
import json
from webapp.entity_layer.encryption.encrypt_confidential_data import EncryptData


def read_params(file_path):
    try:
        with open(file_path) as f:
            configuration = yaml.safe_load(f)
        return configuration
    except Exception as e:
        raise e


def save_configuration(configuration, file_name):
    try:
        with open(file_name, 'w') as f:
            yaml.dump(configuration, f)
        return True
    except Exception as e:
        raise e


if __name__ == "__main__":
    FILE_NAME = "project_config.yaml"
    CONFIG_DETAIL = read_params(FILE_NAME)
    ENCRYPTER = EncryptData()
    KEY = ENCRYPTER.generate_key()
    USER_NAME, PASSWORD, URL = CONFIG_DETAIL['mongodb']['user_name'], \
        CONFIG_DETAIL['mongodb']['password'], CONFIG_DETAIL['mongodb']['url']
    ENCRYPT_USER_NAME = ENCRYPTER.encrypt_message(USER_NAME, KEY)
    ENCRYPT_PASSWORD = ENCRYPTER.encrypt_message(PASSWORD, KEY)
    ENCRYPT_URL = ENCRYPTER.encrypt_message(URL, KEY)
    PROJECT_CREDENTIALS = {
        "key": KEY,
        "mongodb": {
            "user_name": ENCRYPT_USER_NAME,
            "password": ENCRYPT_PASSWORD,
            "url": ENCRYPT_PASSWORD,
            "is_cloud": CONFIG_DETAIL["mongodb"]["is_cloud"]
        },
        "root_folder": CONFIG_DETAIL["root_folder"]
    }
    save_configuration(PROJECT_CREDENTIALS, "project_credentials.yaml")
    from webapp.project_library_layer.credentials.credential_data import save_azure_blob_storage_connection_str, \
        save_aws_credentials, save_google_cloud_storage_credentials, save_watcher_checkpoint_storage_account_connection_str, \
        save_azure_input_file_storage_connection_str, save_user_detail, save_azure_event_hub_namespace_connection_str, \
        save_flask_session_key, save_email_configuration

    CLOUD_STORAGE = CONFIG_DETAIL["cloud_storage"]

    # Azure
    AZURE_CONNECTION_STR = CLOUD_STORAGE['azure_blob_storage']['connection_str']
    save_azure_blob_storage_connection_str(connection_str=AZURE_CONNECTION_STR)

    # Amazon
    AWS_S3_BUCKET_CREDENTIALS = CLOUD_STORAGE['aws_s3_bucket']
    save_aws_credentials(data=AWS_S3_BUCKET_CREDENTIALS)

    # gcp
    GCP_JSON_PATH = CLOUD_STORAGE["gcp"]
    GCP_DETAIL = dict(json.load(open(GCP_JSON_PATH)))
    save_google_cloud_storage_credentials(GCP_DETAIL)

    WATCHER_STORAGE_ACCOUNT = CONFIG_DETAIL["watcher_checkpoint_storage_account_connection_str"]["connection_str"]
    AZURE_INPUT_FILE_CONNECTION_STR = CONFIG_DETAIL[
        "azure_input_file_storage_connection_str"]["connection_str"]
    EVENT_HUB_NAMESPACE_CONNECTION_STR = CONFIG_DETAIL["event_hub_name_space"]["connection_str"]

    save_watcher_checkpoint_storage_account_connection_str(
        WATCHER_STORAGE_ACCOUNT)

    save_azure_input_file_storage_connection_str(
        AZURE_INPUT_FILE_CONNECTION_STR)

    save_azure_event_hub_namespace_connection_str(
        EVENT_HUB_NAMESPACE_CONNECTION_STR)

    # flask session
    SECRET_KEY = CONFIG_DETAIL['session']['secret-key']
    save_flask_session_key(SECRET_KEY)

    USER_DETAIL = CONFIG_DETAIL['user_detail']
    EMAIL = USER_DETAIL["email_id"]
    ROLE_ID = USER_DETAIL['role_id']
    save_user_detail(EMAIL, ROLE_ID)
    EMAIL_CONFIG = CONFIG_DETAIL["email_config"]
    EMAIL_CONFIG["passkey"] = ENCRYPTER.encrypt_message(
        EMAIL_CONFIG["passkey"], KEY)
    save_email_configuration(EMAIL_CONFIG)
    print(CONFIG_DETAIL)
