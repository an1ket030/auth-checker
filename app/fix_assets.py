from PIL import Image, ImageDraw
import os

ASSETS_DIR = "d:/Projects/auth-checker/app/assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

def create_icon(filename, color, size):
    img = Image.new('RGB', size, color=color)
    d = ImageDraw.Draw(img)
    # Draw a simple circle or text to make it look like an icon
    d.ellipse((size[0]//4, size[1]//4, size[0]*3//4, size[1]*3//4), fill="white")
    path = os.path.join(ASSETS_DIR, filename)
    img.save(path, "PNG")
    print(f"Created {filename} at {path}")

# Generate standard assets
create_icon("icon.png", "#4F46E5", (1024, 1024))
create_icon("splash.png", "#4F46E5", (1242, 2436))
create_icon("adaptive-icon.png", "#4F46E5", (1024, 1024))
create_icon("favicon.png", "#4F46E5", (48, 48))
