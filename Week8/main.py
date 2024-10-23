import convolution
from PIL import Image
import numpy as np
import os
import week8

downloads_folder = os.path.expanduser('~/Downloads')
image_files = [f for f in os.listdir(downloads_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

if not image_files:
    raise FileNotFoundError("No image files found in the downloads folder.")

first_image_path = os.path.join(downloads_folder, image_files[0])
im = Image.open(first_image_path)

width, height = im.size
new_height = 200
new_width = int((new_height / height) * width)
im = im.resize((new_width, new_height), Image.LANCZOS)
rgb = np.array(im.convert('RGB'))

r = rgb[:, :, 0]

kernel1 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
kernel2 = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])

result1 = convolution.convolve2d(r, kernel1)
result2 = convolution.convolve2d(r, kernel2)

images_folder = os.path.expanduser('Images')
os.makedirs(images_folder, exist_ok=True)

result1_image_path = os.path.join(images_folder, 'result1.png')
result2_image_path = os.path.join(images_folder, 'result2.png')

Image.fromarray(np.uint8(result1)).save(result1_image_path)
Image.fromarray(np.uint8(result2)).save(result2_image_path)

week8.run_models()