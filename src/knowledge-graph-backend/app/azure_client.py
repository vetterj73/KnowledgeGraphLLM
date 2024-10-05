# app/azure_client.py
from azure.storage.blob.aio import BlobServiceClient
from azure.identity import DefaultAzureCredential
from .config import settings


class AzureBlobClient:
    def __init__(self):
        credential: DefaultAzureCredential = DefaultAzureCredential()
        self.blob_service_client = BlobServiceClient(
            account_url=settings.AZURE_CONNECTION_STRING,
            credential=credential
        )
        self.container_client = self.blob_service_client.get_container_client(
            settings.AZURE_CONTAINER_NAME
        )

    async def upload_file(self, filename: str, data: bytes):
        blob_client = self.container_client.get_blob_client(filename)
        await blob_client.upload_blob(data, overwrite=True)

    async def list_blobs(self):
        return [blob async for blob in self.container_client.list_blobs()]

    async def download_blob(self, filename: str):
        blob_client = self.container_client.get_blob_client(filename)
        stream = await blob_client.download_blob()
        return await stream.readall()
