"""
FRCNN --- Object Detection
This is the code to do object detection (Part 1).
The total code has three parts: Part 1 image perparion. Part 2 train model. Part 3 test model
I used the Open Images V5 data, Keras with tensorflow.
The range of the bounding box is x: (0,1) and y: (0,1)
07/27/2020
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import random
from skimage import io
from shutil import copyfile
import sys
import time

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# build parameters
images_boxable_fname = '/Users/Darren/Desktop/OD_ml/train-images-boxable-with-rotation.csv'
annotations_bbox_fname = '/Users/Darren/Desktop/OD_ml/train-annotations-bbox.csv'
class_descriptions_fname = '/Users/Darren/Desktop/OD_ml/class-descriptions-boxable.csv'

images_boxable = pd.read_csv(images_boxable_fname)  # get label code and url link
print(images_boxable.head())

annotations_bbox = pd.read_csv(annotations_bbox_fname)  # get bounding box cooridate and label code
print(annotations_bbox.head())

class_descriptions = pd.read_csv(class_descriptions_fname, header=None)  # get label name and lable code
print(class_descriptions.head())

# plot bounding box comment to speed up
def plot_bbox(img_id):
  img_url = images_boxable.loc[images_boxable["ImageID"]==img_id]['OriginalURL'].values[0]
  img = io.imread(img_url)
  height, width, channel = img.shape
  print(f"Image: {img.shape}")
  bboxs = annotations_bbox[annotations_bbox['ImageID']==img_id]
  for index, row in bboxs.iterrows():
      xmin = row['XMin']
      xmax = row['XMax']
      ymin = row['YMin']
      ymax = row['YMax']
      xmin = int(xmin*width)
      xmax = int(xmax*width)
      ymin = int(ymin*height)
      ymax = int(ymax*height)
      label_name = row['LabelName']
      class_series = class_descriptions[class_descriptions[0]==label_name]
      class_name = class_series[1].values[0]
      print(f"Coordinates: {xmin,ymin}, {xmax,ymax}")
      cv2.rectangle(np.array(img), (xmin,ymin), (xmax,ymax), (255,0,0), 5)     ### np.array(img) or img
      # annotate image with text (optional)
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(img, class_name, (xmin,ymin-10), font, 3, (0,255,0), 5)
  plt.figure(figsize=(15,10))
  # plt.title('Image with Bounding Box')
  plt.imshow(img)
  plt.axis("off")
  plt.show()

# # This part of code is for ploting image. Comment to save time.
# # limit the numbers of objects in each image to make it easily visualization
# least_objects_img_ids = annotations_bbox["ImageID"].value_counts().tail(50).index.values
# for img_id in random.sample(list(least_objects_img_ids), 3):
#   plot_bbox(img_id)

# Find the label_name for 'Apple' and 'Bee' classes
apple_pd = class_descriptions[class_descriptions[1]=='Apple']
bee_pd = class_descriptions[class_descriptions[1]=='Bee']
label_name_apple = apple_pd[0].values[0]
label_name_bee = bee_pd[0].values[0]

# check the total number of each class and choose unique one
apple_bbox = annotations_bbox[annotations_bbox['LabelName']==label_name_apple]
bee_bbox = annotations_bbox[annotations_bbox['LabelName']==label_name_bee]
print('There are %d apple in the dataset' %(len(apple_bbox)))
print('There are %d bee in the dataset' %(len(bee_bbox)))
apple_img_id = apple_bbox['ImageID']
bee_img_id = bee_bbox['ImageID']
# get the unique box in one image
apple_img_id = np.unique(apple_img_id)
bee_img_id = np.unique(bee_img_id)
print('There are %d images which contain apple' % (len(apple_img_id)))
print('There are %d images which contain bee' % (len(bee_img_id)))

# To speed up, I just choose 24 images/class
n = 24
subapple_img_id = random.sample(list(apple_img_id), n)
subbee_img_id = random.sample(list(bee_img_id), n)

subapple_pd = images_boxable.loc[images_boxable['ImageID'].isin(subapple_img_id)]
subbee_pd = images_boxable.loc[images_boxable['ImageID'].isin(subbee_img_id)]

subapple_dict = subapple_pd[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()
subbee_dict = subbee_pd[["ImageID", "OriginalURL"]].set_index('ImageID')["OriginalURL"].to_dict()
mappings = [subapple_dict, subbee_dict]
print(len(mappings[0]))
classes = ['Apple', 'Bee']

# download images and save for further steps
for idx, obj_type in enumerate(classes):
  n_issues = 0
  # create the directory
  if not os.path.exists(obj_type):
    os.mkdir(obj_type)
  for img_id, url in mappings[idx].items():
    try:
      img = io.imread(url)
      saved_path = os.path.join(obj_type, img_id+".jpg")
      io.imsave(saved_path, img)
    except Exception as e:
      n_issues += 1
  print(f"Images Issues: {n_issues}")

# Above codes is for working on the training dataset and save to the class name folder.
# working on the data to fit the Faster-RCNN format (fname_path, xmin, xmax, ymin ymax, class_name)
# save images to train path and test path by creating two folders in your main path
train_path = 'train'
test_path = 'test'
random.seed(10)
for i in range(len(classes)):
  all_imgs = os.listdir(classes[i])
  all_imgs = [f for f in all_imgs if not f.startswith('.')]
  random.shuffle(all_imgs)
  limit = int(n*0.8)   # with training data 80%, and test data 20%.

  train_imgs = all_imgs[:limit]
  test_imgs = all_imgs[limit:]

  # copy each classes' images to train directory
  for j in range(len(train_imgs)):
    original_path = os.path.join(classes[i], train_imgs[j])
    new_path = os.path.join(train_path, train_imgs[j])
    copyfile(original_path, new_path)

  # copy each classes' images to test directory
  for j in range(len(test_imgs)):
    original_path = os.path.join(classes[i], test_imgs[j])
    new_path = os.path.join(test_path, test_imgs[j])
    copyfile(original_path, new_path)

# some images have more than one class. [DON'T UNDERSTAND THIS PART]
label_names = [label_name_apple, label_name_bee]
train_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])

# Find boxes in each image and put them in a dataframe
# for trianing data
train_imgs = os.listdir(train_path)
train_imgs = [name for name in train_imgs if not name.startswith('.')]
for i in range(len(train_imgs)):
    sys.stdout.write('Parse train_imgs ' + str(i) + '; Number of boxes: ' + str(len(train_df)) + '\r')
    sys.stdout.flush()
    img_name = train_imgs[i]
    img_id = img_name[0:16]   # [0:16] is the charactor of image name
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(label_names)):
            if labelName == label_names[i]:
                train_df = train_df.append({'FileName': img_name,
                                            'XMin': row['XMin'],
                                            'XMax': row['XMax'],
                                            'YMin': row['YMin'],
                                            'YMax': row['YMax'],
                                            'ClassName': classes[i]},
                                           ignore_index=True)
print(train_df.head())
train_img_ids = train_df["FileName"].head().str.split(".").str[0].unique()
for img_id in train_img_ids:
  plot_bbox(img_id)

# for test data
test_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
# find boxes in each image and put them in a dataframe
test_imgs = os.listdir(test_path)
test_imgs = [name for name in test_imgs if not name.startswith('.')]
for i in range(len(test_imgs)):
    sys.stdout.write('Parse test_imgs ' + str(i) + '; Number of boxes: ' + str(len(test_df)) + '\r')
    sys.stdout.flush()
    img_name = test_imgs[i]
    img_id = img_name[0:16]
    tmp_df = annotations_bbox[annotations_bbox['ImageID']==img_id]
    for index, row in tmp_df.iterrows():
        labelName = row['LabelName']
        for i in range(len(label_names)):
            if labelName == label_names[i]:
                test_df = test_df.append({'FileName': img_name,
                                            'XMin': row['XMin'],
                                            'XMax': row['XMax'],
                                            'YMin': row['YMin'],
                                            'YMax': row['YMax'],
                                            'ClassName': classes[i]},
                                           ignore_index=True)

train_df.to_csv('train.csv')
test_df.to_csv('test.csv')

# last step: write these csv file into a annotation.txt file.
# for training
train_df = pd.read_csv('train.csv')
with open("annotation.txt", "w+") as f:
    for idx, row in train_df.iterrows():
        img = cv2.imread('train/' + row['FileName'])
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)

        file_path = '/Users/Darren/Desktop/OD_ml/train'
        fileName = os.path.join(file_path, row['FileName'])
        className = row['ClassName']
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')

# for test
test_df = pd.read_csv('test.csv')
with open("test_annotation.txt", "w+") as f:
    for idx, row in test_df.iterrows():
        sys.stdout.write(str(idx) + '\r')
        sys.stdout.flush()
        img = cv2.imread('test/' + row['FileName'])
        height, width = img.shape[:2]
        x1 = int(row['XMin'] * width)
        x2 = int(row['XMax'] * width)
        y1 = int(row['YMin'] * height)
        y2 = int(row['YMax'] * height)

        file_path = '/Users/Darren/Desktop/OD_ml/test'
        fileName = os.path.join(file_path, row['FileName'])
        className = row['ClassName']
        f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + className + '\n')