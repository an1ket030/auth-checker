import os
import uuid
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Cloudinary Configuration
# Documentation: https://cloudinary.com/documentation/django_integration
CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")

is_configured = False

if CLOUDINARY_URL:
    try:
        # cloudinary package will automatically pick up the CLOUDINARY_URL env var
        # if we just import it, but we can also explicitly configure it if needed.
        # Since it's in the environment, it auto-configures.
        is_configured = True
        print("[Storage] Cloudinary client initialized successfully from CLOUDINARY_URL.")
    except Exception as e:
        print(f"[Storage] Failed to initialize Cloudinary: {e}")
else:
    print("[Storage] WARNING: CLOUDINARY_URL not found. Image uploads will fallback to local storage.")


def upload_image_to_r2(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """
    Uploads an image to Cloudinary and returns the public URL.
    (Kept the function name 'upload_image_to_r2' for backwards compatibility with server.py, 
    but it now uploads to Cloudinary).
    """
    if not is_configured:
        print("[Storage] WARNING: Upload skipped. Cloudinary is not configured.")
        return None

    try:
        # Generate unique filename: uuid
        # (Cloudinary handles extensions automatically)
        unique_filename = uuid.uuid4().hex

        # Upload to Cloudinary
        response = cloudinary.uploader.upload(
            file_bytes,
            public_id=f"auth_checker/scans/{unique_filename}",
            resource_type="image"
        )
        
        # Construct the public URL (using secure URL)
        url = response.get('secure_url')
            
        print(f"[Storage] Successfully uploaded to Cloudinary: {url}")
        return url
        
    except Exception as e:
        print(f"[Storage Error] Unexpected error during Cloudinary upload: {e}")
        return None


def delete_image_from_r2(image_url: str) -> bool:
    """
    Deletes an object from Cloudinary given its public URL.
    """
    if not is_configured or not image_url:
        return False
        
    try:
        # Extract the public_id from the URL
        # Example URL: https://res.cloudinary.com/demo/image/upload/v1234567890/auth_checker/scans/abcdef.jpg
        # We need "auth_checker/scans/abcdef"
        
        # Split by /upload/
        parts = image_url.split('/upload/')
        if len(parts) > 1:
            path_part = parts[1]
            # Remove version number if present (e.g., v12345/, sometimes it's not present)
            path_parts = path_part.split('/')
            if path_parts[0].startswith('v') and path_parts[0][1:].isdigit():
                path_parts.pop(0)
            
            # Rejoin and remove extension
            key_with_ext = '/'.join(path_parts)
            public_id = os.path.splitext(key_with_ext)[0]
            
            cloudinary.uploader.destroy(public_id)
            print(f"[Storage] Successfully deleted {public_id} from Cloudinary")
            return True
            
        return False
    except Exception as e:
        print(f"[Storage Error] Unexpected error during Cloudinary delete: {e}")
        return False
