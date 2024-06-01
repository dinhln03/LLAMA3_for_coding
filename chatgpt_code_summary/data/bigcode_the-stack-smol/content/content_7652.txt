from datetime import datetime, timedelta
from uuid import uuid4


class Uploader:
    def generate_token(self):
        pass

    def generate_download_link(self, object_name, filename) -> (dict, str):
        pass

    def object_name(self) -> str:
        return str(uuid4())


class MockUploader(Uploader):
    def __init__(self, config):
        self.config = config

    def get_token(self):
        return ({}, self.object_name())

    def generate_download_link(self, object_name, filename):
        return ""


class AzureUploader(Uploader):
    def __init__(self, config):
        self.account_name = config["AZURE_ACCOUNT_NAME"]
        self.storage_key = config["AZURE_STORAGE_KEY"]
        self.container_name = config["AZURE_TO_BUCKET_NAME"]
        self.timeout = timedelta(seconds=config["PERMANENT_SESSION_LIFETIME"])

        from azure.storage.common import CloudStorageAccount
        from azure.storage.blob import BlobPermissions

        self.CloudStorageAccount = CloudStorageAccount
        self.BlobPermissions = BlobPermissions

    def get_token(self):
        """
        Generates an Azure SAS token for pre-authorizing a file upload.

        Returns a tuple in the following format: (token_dict, object_name), where
            - token_dict has a `token` key which contains the SAS token as a string
            - object_name is a string
        """
        account = self.CloudStorageAccount(
            account_name=self.account_name, account_key=self.storage_key
        )
        bbs = account.create_block_blob_service()
        object_name = self.object_name()
        sas_token = bbs.generate_blob_shared_access_signature(
            self.container_name,
            object_name,
            permission=self.BlobPermissions.CREATE,
            expiry=datetime.utcnow() + self.timeout,
            protocol="https",
        )
        return ({"token": sas_token}, object_name)

    def generate_download_link(self, object_name, filename):
        account = self.CloudStorageAccount(
            account_name=self.account_name, account_key=self.storage_key
        )
        bbs = account.create_block_blob_service()
        sas_token = bbs.generate_blob_shared_access_signature(
            self.container_name,
            object_name,
            permission=self.BlobPermissions.READ,
            expiry=datetime.utcnow() + self.timeout,
            content_disposition=f"attachment; filename={filename}",
            protocol="https",
        )
        return bbs.make_blob_url(
            self.container_name, object_name, protocol="https", sas_token=sas_token
        )
