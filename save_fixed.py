import os
from argparse import ArgumentParser
from utils import draw_bboxes
import cv2 


parser = ArgumentParser()
parser.add_argument('--input_dir', required=True, help="Directory in which annotated images should be moved")
parser.add_argument('--save_empty', required=False, help="Indicates if images without annotation should be moved or not")
args = parser.parse_args()

cwd = args.input_dir
save_empty = args.save_empty

fixed_dir = os.path.join(cwd, "fixed")
skipped_images_dir = os.path.join(cwd, "skipped_images")
original_images_dir = os.path.join(cwd, "images")
annotated_images_dir = os.path.join(cwd, "annotated")
labels_dir = os.path.join(cwd, "labels")

for filename in os.listdir(fixed_dir):
    if filename != "classes.txt":
        with open(os.path.join(fixed_dir, filename), "r") as f:
            bbox = f.read().split(' ')[1:]
            bbox = [float(coordinate) for coordinate in bbox]
        image_filename = filename[:filename.rfind('.')] + ".jpg"
        image = cv2.imread(os.path.join(skipped_images_dir, image_filename))
        annotated_image = draw_bboxes(bbox, image)
        os.rename(os.path.join(skipped_images_dir, image_filename), os.path.join(original_images_dir, image_filename)) # Move image to another directory
        os.rename(os.path.join(fixed_dir, filename), os.path.join(labels_dir, filename)) # Move label file to another directory
        
        print("Image", image_filename, "saved in dir:", annotated_images_dir)
        cv2.imwrite(os.path.join(annotated_images_dir, image_filename), annotated_image)
        
if save_empty:
    for image_filename in os.listdir(skipped_images_dir):
        os.rename(os.path.join(skipped_images_dir, image_filename), os.path.join(original_images_dir, image_filename)) # Move image to another directory
        label_filename = image_filename[:image_filename.rfind('.')] + ".txt"
        print(label_filename)
        with open(os.path.join(labels_dir, label_filename), "w+") as f:
            f.write('')