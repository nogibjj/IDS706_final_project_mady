import os
import boto3

BUCKET_NAME = "mady-ids706-final-proj"
LOCAL_FOLDER = "../data"

# Initialize the S3 client
s3_client = boto3.client("s3")

# Iterate through files in the folder and upload each one
for root, dirs, files in os.walk(LOCAL_FOLDER):
    for file_name in files:
        # Construct the full local path
        local_file_path = os.path.join(root, file_name)

        # Define the S3 object key (path in the bucket)
        s3_object_key = os.path.relpath(local_file_path, LOCAL_FOLDER).replace(
            "\\", "/"
        )  # Ensure consistency across OSes

        try:
            # Upload the file
            s3_client.upload_file(local_file_path, BUCKET_NAME, s3_object_key)
            print(f"Uploaded: {local_file_path} to s3://{BUCKET_NAME}/{s3_object_key}")
        except Exception as e:
            print(f"Error uploading {local_file_path}: {e}")
