import os
import boto3
from botocore.exceptions import ClientError
from fastapi import UploadFile
import uuid

# Cloudflare R2 Configuration
# Documentation: https://developers.cloudflare.com/r2/api/s3/api/
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "auth-checker-images")
R2_PUBLIC_URL_PREFIX = os.getenv("R2_PUBLIC_URL_PREFIX") # e.g., https://pub-xxxxxxxxxx.r2.dev

s3_client = None

def get_s3_client():
    global s3_client
    if s3_client is None and R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
                aws_access_key_id=R2_ACCESS_KEY_ID,
                aws_secret_access_key=R2_SECRET_ACCESS_KEY,
                region_name='auto' # Cloudflare R2 requires 'auto'
            )
            print("[Storage] Cloudflare R2 client initialized successfully.")
        except Exception as e:
            print(f"[Storage] Failed to initialize R2 client: {e}")
    return s3_client


def upload_image_to_r2(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """
    Uploads an image to Cloudflare R2 and returns the public URL.
    Generates a unique filename using UUID to prevent collisions.
    """
    client = get_s3_client()
    
    # If R2 is not configured, fallback to returning None or error
    if not client:
        print("[Storage] WARNING: Upload skipped. R2 is not configured.")
        return None

    try:
        # Generate unique filename: uuid + extension
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            ext = ".jpg"
        unique_filename = f"{uuid.uuid4().hex}{ext}"

        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=unique_filename,
            Body=file_bytes,
            ContentType=content_type
        )
        
        # Construct the public URL
        if R2_PUBLIC_URL_PREFIX:
            url = f"{R2_PUBLIC_URL_PREFIX.rstrip('/')}/{unique_filename}"
        else:
            # Fallback format if public prefix is not set, though typical R2 uses custom domains or r2.dev
            url = f"https://{R2_BUCKET_NAME}.{R2_ACCOUNT_ID}.r2.cloudflarestorage.com/{unique_filename}"
            
        print(f"[Storage] Successfully uploaded {unique_filename} to R2")
        return url
        
    except ClientError as e:
        print(f"[Storage Error] Boto3 client error during upload: {e}")
        return None
    except Exception as e:
        print(f"[Storage Error] Unexpected error during upload: {e}")
        return None


def delete_image_from_r2(image_url: str) -> bool:
    """
    Deletes an object from Cloudflare R2 given its public URL.
    """
    client = get_s3_client()
    if not client or not image_url:
        return False
        
    try:
        # Extract the key (filename) from the URL
        if R2_PUBLIC_URL_PREFIX and image_url.startswith(R2_PUBLIC_URL_PREFIX):
            key = image_url.replace(R2_PUBLIC_URL_PREFIX.rstrip('/') + '/', "")
        else:
            # Try to grab the last part of the URL
            key = image_url.split('/')[-1]

        client.delete_object(
            Bucket=R2_BUCKET_NAME,
            Key=key
        )
        print(f"[Storage] Successfully deleted {key} from R2")
        return True
    except ClientError as e:
        print(f"[Storage Error] Boto3 client error during delete: {e}")
        return False
    except Exception as e:
        print(f"[Storage Error] Unexpected error during delete: {e}")
        return False
