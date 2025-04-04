import os
import torch
import random
import copy
import csv
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import numpy as np
import pydicom as dicom
import cv2
from skimage import transform, io, img_as_float, exposure
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, RandomResizedCrop, Normalize
)
from albumentations.pytorch import ToTensorV2

def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_ts_transformations(crop_size):
    AUGMENTATIONS = Compose([
      RandomResizedCrop(height=crop_size, width=crop_size),
      ShiftScaleRotate(rotate_limit=10),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
    ])
    return AUGMENTATIONS


class ChestXray14(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


# ---------------------------------------------Downstream CheXpert------------------------------------------
class CheXpert(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        if "test" in line[0]:
          label = line[1:]
        else:
          label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream Shenzhen------------------------------------------
class ShenzhenCXR(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=1, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)

    with open(file_path, "r") as fr:
      line = fr.readline().strip()
      while line:
        lineItems = line.split()
        imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
        imageLabel = lineItems[1:]
        imageLabel = [int(i) for i in imageLabel]
        self.img_list.append(imagePath)
        self.img_label.append(imageLabel)
        line = fr.readline()

    if annotation_percent < 100:
      indexes = np.arange(len(self.img_list))
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel
    
  def __len__(self):

    return len(self.img_list)

#---3 rows VinDrCXR ---
class VinDrCXR_row1(VinDrCXR):
    def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
        super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_row2(VinDrCXR):
    def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
        super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_row3(VinDrCXR):
    def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
        super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

#--- 17 radiologists VinDrCXR ---
class VinDrCXR_rad1(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad2(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad3(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad4(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad5(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad6(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad7(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad8(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad9(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad10(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad11(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad12(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad13(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad14(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad15(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad16(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

class VinDrCXR_rad17(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
# ----- Randomly distributed VinDrCXR data, using 17 radiologists distribution -----
class VinDrCXR_random1(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random2(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random3(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random4(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random5(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random6(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random7(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random8(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random9(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random10(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random11(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random12(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random13(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random14(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random15(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random16(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)
    
class VinDrCXR_random17(VinDrCXR):
  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=6, annotation_percent=100):
    super().__init__(images_path, file_path, crop_size, resize, augment, num_class, annotation_percent)

# ---------------------------------------------Downstream RSNA Pneumonia------------------------------------------
class RSNAPneumonia(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])


          self.img_list.append(imagePath)
          imageLabel = np.zeros(3)
          imageLabel[int(lineItems[-1])] = 1
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

class MIMIC(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=14,
               uncertain_label="Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.crop_size = crop_size
    self.resize = resize
 
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB').resize((self.resize,self.resize))
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      teacher_img=np.array(imageData.resize((self.crop_size,self.crop_size))) / 255.
      
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData)
      student_img = augmented['image']
      student_img=np.array(student_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)

class COVIDx(Dataset):

  def __init__(self, images_path, file_path, crop_size=224, resize=256, augment=None, num_class=3, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations(crop_size)
    classes = ['normal', 'pneumonia', 'COVID-19']
    
    images_path = os.path.join(images_path, 'train') if 'train' in file_path else os.path.join(images_path, 'test')
    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          patient_id, fname, label, source  = line.strip().split(' ')
          imagePath = os.path.join(images_path, fname)

          self.img_list.append(imagePath)
          
          imageLabel = np.zeros(3)
          imageLabel[classes.index(label)] = 1
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    cv2.setNumThreads(0)
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')

    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


dict_dataloarder = {
  "ChestXray14": ChestXray14,
  "CheXpert": CheXpert,
  "Shenzhen": ShenzhenCXR,
  "VinDrCXR": VinDrCXR,
  "RSNAPneumonia": RSNAPneumonia,
  "MIMIC": MIMIC,
  "COVIDx": COVIDx,
  "VinDrCXR_row1": VinDrCXR_row1,
  "VinDrCXR_row2": VinDrCXR_row2,
  "VinDrCXR_row3": VinDrCXR_row3,
  "VinDrCXR_rad1": VinDrCXR_rad1,
  "VinDrCXR_rad2": VinDrCXR_rad2,
  "VinDrCXR_rad3": VinDrCXR_rad3,
  "VinDrCXR_rad4": VinDrCXR_rad4,
  "VinDrCXR_rad5": VinDrCXR_rad5,
  "VinDrCXR_rad6": VinDrCXR_rad6,
  "VinDrCXR_rad7": VinDrCXR_rad7,
  "VinDrCXR_rad8": VinDrCXR_rad8,
  "VinDrCXR_rad9": VinDrCXR_rad9,
  "VinDrCXR_rad10": VinDrCXR_rad10,
  "VinDrCXR_rad11": VinDrCXR_rad11,
  "VinDrCXR_rad12": VinDrCXR_rad12,
  "VinDrCXR_rad13": VinDrCXR_rad13,
  "VinDrCXR_rad14": VinDrCXR_rad14,
  "VinDrCXR_rad15": VinDrCXR_rad15,
  "VinDrCXR_rad16": VinDrCXR_rad16,
  "VinDrCXR_rad17": VinDrCXR_rad17,
  "VinDrCXR_random1": VinDrCXR_random1,
  "VinDrCXR_random2": VinDrCXR_random2,
  "VinDrCXR_random3": VinDrCXR_random3,
  "VinDrCXR_random4": VinDrCXR_random4,
  "VinDrCXR_random5": VinDrCXR_random5,
  "VinDrCXR_random6": VinDrCXR_random6,
  "VinDrCXR_random7": VinDrCXR_random7,
  "VinDrCXR_random8": VinDrCXR_random8,
  "VinDrCXR_random9": VinDrCXR_random9,
  "VinDrCXR_random10": VinDrCXR_random10,
  "VinDrCXR_random11": VinDrCXR_random11,
  "VinDrCXR_random12": VinDrCXR_random12,
  "VinDrCXR_random13": VinDrCXR_random13,
  "VinDrCXR_random14": VinDrCXR_random14,
  "VinDrCXR_random15": VinDrCXR_random15,
  "VinDrCXR_random16": VinDrCXR_random16,
  "VinDrCXR_random17": VinDrCXR_random17
}
