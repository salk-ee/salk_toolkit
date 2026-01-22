"""Util to upload HTML graph to public S3 bucket"""

import hashlib
import logging
from warnings import warn

import boto3
from botocore.exceptions import ClientError

# S3 configuration
S3_BUCKET = "salk-public"
S3_DIRECTORY = "graph-html/"


def upload_html_to_s3(html_str: str) -> str | None:
    """Upload an HTML string to S3 bucket with checksum validation.

    Credentials are automatically loaded by boto3 from ~/.aws/credentials or environment variables.

    :param html_str: HTML string to be uploaded
    :return: Public URL to the uploaded HTML file, or None if upload failed
    """
    # Generate filename from content hash if not provided
    content_hash = hashlib.sha256(html_str.encode()).hexdigest()[:16]
    filename = f"{content_hash}.html"

    # Ensure .html extension
    if not filename.endswith(".html"):
        filename = f"{filename}.html"

    # Construct full S3 key
    s3_key = f"{S3_DIRECTORY}{filename}"

    # Construct public URL
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"

    # Create S3 client (uses default credential chain: ~/.aws, environment variables, IAM role, etc.)
    s3_client = boto3.client("s3")

    try:
        # Check if object exists
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        warn(f"File {s3_key} already exists in the bucket. Skipping upload.")
        return public_url

    except ClientError as e:
        # If 404, it doesn't exist, so we proceed to upload
        error_code = e.response.get("Error", {}).get("Code")
        if error_code != "404":
            logging.error(f"Error checking object {s3_key}: {e}")
            return None

    # Upload HTML string as bytes
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=html_str.encode("utf-8"),
            ContentType="text/html",
            ChecksumAlgorithm="SHA256",
        )
        logging.info(f"Successfully uploaded {s3_key}")
        return public_url
    except ClientError as e:
        logging.error(f"Failed to upload {s3_key}: {e}")
        return None
