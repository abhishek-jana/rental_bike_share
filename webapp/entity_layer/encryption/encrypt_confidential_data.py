import os
import sys
from passlib.hash import pbkdf2_sha256
from cryptography.fernet import Fernet
from webapp.exception_layer.encryption_exception.encryption_exception import EncryptionException


class EncryptData:
    """
    class to encrypt data
    """

    def __init__(self):
        pass

    def get_encrypted_text(self, text):
        """
        This function will return hash calculated on your data
        :param data:
        :return encrypted hash:
        """

        try:
            # start_date = date_time.get_date()
            # start_time = date_time.get_time()
            if text is not None:
                hash_data = pbkdf2_sha256.hash(text)
                return hash_data

        except Exception as e:
            encryption_exception = EncryptionException("Error occurred in module [{0}] class [{1}] method [{2}]"
                                                       .format(self.__module__, EncryptData.__name__, self.__init__.__name__))
            raise Exception(
                encryption_exception.error_message_detail(str(e), sys)) from e

    def verify_encrypted_text(self, text, encrypted_text):
        try:
            return pbkdf2_sha256.verify(text, encrypted_text)
        except Exception as e:
            raise e

    def generate_key(self):
        """
        generate key and save it into as file
        """
        key = Fernet.generate_key()
        key = key.decode('utf-8')
        return key

    def load_key(self):
        """
        load key from mongodb env
        """
        key = os.environ.get('SECRET_KEY_MONGO_DB', None)
        key = key.encode('utf-8')
        return key

    def encrypt_message(self, message, key=None):
        """
        encrypts a message
        """
        encoded_message = message.encode()

        if key is None:
            key = self.load_key()
        f = Fernet(key)
        encrypted_message = f.encrypt(encoded_message)

        return encrypted_message

    def decrypt_message(self, encrypted_message, key=None):
        """
        Decrypts a message
        """
        if key is None:
            import yaml
            config = yaml.safe_load(open("project_credentials.yaml"))
            key = config["key"]
        f = Fernet(key)
        decrypted_message = f.decrypt(encrypted_message)
        return decrypted_message
