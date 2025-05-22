import os
import argparse
from PIL import Image

def resize_images(directory, width, height):
    print(f'Processing directory: {directory} with target size: {width}x{height}')
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            print(f'Resizing image: {filename}')
            with Image.open(file_path) as img:
                resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                resized_img.save(file_path)
            print(f'Resized image: {filename}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resize images in a directory.')
    parser.add_argument('directory', type=str, help='Directory containing images to resize')
    parser.add_argument('width', type=int, help='Target width for resized images')
    parser.add_argument('height', type=int, help='Target height for resized images')
    args = parser.parse_args()
    resize_images(args.directory, args.width, args.height)
