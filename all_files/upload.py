import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

try:
    print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")

    # Quick start code goes here

except Exception as ex:
    print('Exception:')
    print(ex)

connect_str = "DefaultEndpointsProtocol=https;AccountName=asrdataprocessed;AccountKey=dha6CnVEzlodiz8niBwSQIIAgGjxgJzGj/STITJGVxVy4Bnl6mlmHRvcoD11c30O4cS3cEaJhs9hGVFCIgJ9uA==;EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Create a local directory to hold blob data
local_path = "/workspace/data_format_scrips.zip"
# os.mkdir(local_path)

# Create a file in the local data directory to upload and download
local_file_name = "data_format_scrips.zip"
upload_file_path = "/workspace/data_format_scrips.zip"

# # Write text to the file
# file = open(upload_file_path, 'w')
# file.write("Hello, World!")
# file.close()

# Create a blob client using the local file name as the name for the blob
blob_client = blob_service_client.get_blob_client(container="data", blob="data_format_scrips.zip")

print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

# Upload the created file
with open(upload_file_path, "rb") as data:
    blob_client.upload_blob(data)