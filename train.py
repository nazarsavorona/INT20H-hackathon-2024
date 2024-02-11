from ultralytics import YOLO
import pandas as pd
import os
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydicom import dcmread
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.utils import save_image

import matplotlib.patches as patches
from math import ceil

# --------------------------------- Preparing the data ---------------------------------
DATA_FOLDER = "rsna-pneumonia-detection-challenge/"
random_seed = 42

np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Loading the datasets
label_data = pd.read_csv(DATA_FOLDER + "stage_2_train_labels.csv")
class_info = pd.read_csv(
    DATA_FOLDER + "stage_2_detailed_class_info.csv"
).drop_duplicates()

label_data = label_data.merge(class_info, on="patientId")

columns = ["patientId", "Target"]
all_data = label_data

# Splitting the data into train, validation and test sets
train_labels, test_labels = train_test_split(
    label_data.values, test_size=0.2, random_state=random_seed
)
train_labels, val_labels = train_test_split(
    train_labels, test_size=0.25, random_state=random_seed
)

train_f = DATA_FOLDER + "stage_2_train_images/"

train_paths = [os.path.join(train_f, image[0]) for image in train_labels]
val_paths = [os.path.join(train_f, image[0]) for image in val_labels]
test_paths = [os.path.join(train_f, image[0]) for image in test_labels]

# Transformations
old_size = 1024
new_size = 640

transform = transforms.Compose([transforms.Resize(new_size), transforms.ToTensor()])


class Dataset(data.Dataset):

    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = dcmread(f"{self.paths[index]}.dcm")
        image = image.pixel_array
        image = image / 255.0

        image = (255 * image).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        if self.labels is not None:
            label = self.labels[index][1]
        else:
            label = None

        if self.transform is not None:
            image = self.transform(image)

        name = self.paths[index].split("/")[-1]
        FIL = all_data[all_data["patientId"] == name]

        centers_x = (FIL["x"].values + FIL["width"].values / 2) / old_size
        centers_y = (FIL["y"].values + FIL["height"].values / 2) / old_size

        new_widths = FIL["width"].values / old_size
        new_heights = FIL["height"].values / old_size

        boxes = [
            [centers_x[j], centers_y[j], new_widths[j], new_heights[j]]
            for j in range(len(FIL))
        ]

        return image, label, boxes, name

    def __len__(self):

        return len(self.paths)


# Creating the datasets
train_dataset = Dataset(train_paths, train_labels, transform=transform)
val_dataset = Dataset(val_paths, val_labels, transform=transform)
test_dataset = Dataset(test_paths, test_labels, transform=transform)

# Saving images and labels
data_folder = "datasets/"

# For detection
detection_folder = data_folder + "pneumonia_detection/"

images_folder = detection_folder + "images/"
labels_folder = detection_folder + "labels/"

train_folder = "train/"
val_folder = "val/"
test_folder = "test/"

Path(images_folder + train_folder).mkdir(parents=True, exist_ok=True)
Path(images_folder + val_folder).mkdir(parents=True, exist_ok=True)
Path(images_folder + test_folder).mkdir(parents=True, exist_ok=True)

Path(labels_folder + train_folder).mkdir(parents=True, exist_ok=True)
Path(labels_folder + val_folder).mkdir(parents=True, exist_ok=True)
Path(labels_folder + test_folder).mkdir(parents=True, exist_ok=True)


def save_files_detection(dataset, folder):
    for i, (image, label, boxes, name) in tqdm(enumerate(dataset), total=len(dataset)):
        if label == 1:
            save_image(image, images_folder + folder + f"{name}.png")
            with open(labels_folder + folder + f"{name}.txt", "w") as f:
                for box in boxes:
                    f.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")


save_files_detection(train_dataset, train_folder)
save_files_detection(val_dataset, val_folder)
save_files_detection(test_dataset, test_folder)

# For classification
classification_folder = data_folder + "pneumonia_classification/"

train_folder = "train/"
val_folder = "val/"
test_folder = "test/"

healthy = "0/"
pneumonia = "1/"

Path(classification_folder + train_folder + healthy).mkdir(parents=True, exist_ok=True)
Path(classification_folder + val_folder + healthy).mkdir(parents=True, exist_ok=True)
Path(classification_folder + test_folder + healthy).mkdir(parents=True, exist_ok=True)

Path(classification_folder + train_folder + pneumonia).mkdir(
    parents=True, exist_ok=True
)
Path(classification_folder + val_folder + pneumonia).mkdir(parents=True, exist_ok=True)
Path(classification_folder + test_folder + pneumonia).mkdir(parents=True, exist_ok=True)


def save_files_classification(dataset, folder):
    for i, (image, label, _, name) in tqdm(enumerate(dataset), total=len(dataset)):
        if label == 0:
            save_image(image, classification_folder + folder + healthy + f"{name}.png")
        else:
            save_image(
                image, classification_folder + folder + pneumonia + f"{name}.png"
            )


save_files_classification(train_dataset, train_folder)
save_files_classification(val_dataset, val_folder)
save_files_classification(test_dataset, test_folder)

# --------------------------------- Training the model for classification ---------------------------------
# Model
model_name = "yolov8m-cls.pt"
data_file = "pneumonia_classification.yaml"
best_model_file = "./runs/detect/train8/weights/best.pt"
num_epochs = 20

# model = YOLO(best_model_file)
model = YOLO(model_name)

experiment_name = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{model_name.split(".")[0]}_{data_file}_{num_epochs}epochs'

# Training
results = model.train(
    data=data_file,
    epochs=num_epochs,
    close_mosaic=0,
    dropout=0.25,
    device=0,
    freeze=6,
    optimizer="AdamW",
    lr0=1e-4,
    lrf=1e-3,
    cos_lr=True,
    warmup_epochs=2,
    name=experiment_name,
    erasing=0,
)

success = model.export(format='onnx')

# For submission
submission_folder = data_folder + "pneumonia_inference/"
submission_images_folder = submission_folder + "images/"

Path(submission_images_folder).mkdir(parents=True, exist_ok=True)


def save_submission_files(dataset, folder):
    for i, (image, _, _, name) in tqdm(enumerate(dataset), total=len(dataset)):
        save_image(image, folder + f"{name}.png")


inference_f = DATA_FOLDER + "stage_2_test_images/"

inference_paths = [
    os.path.join(inference_f, image.split(".")[0]) for image in os.listdir(inference_f)
]
inference_dataset = Dataset(inference_paths, None, transform=transform)

save_submission_files(inference_dataset, submission_images_folder)

# --------------------------------- Training the model for detection ---------------------------------
# Model
model_name = "yolov8m.pt"
data_file = "pneumonia_detection.yaml"
best_model_file = "./runs/detect/train8/weights/best.pt"
num_epochs = 20

# model = YOLO(best_model_file)
model = YOLO(model_name)

# Training
results = model.train(data=data_file,
                      epochs=num_epochs,
                      close_mosaic=0,
                      dropout=0.1,
                      device=0,
                      optimizer="AdamW",
                      lr0=1e-3,
                      lrf=1e-2,
                      freeze=10,
                      cos_lr=True,
                      amp=False,
                      warmup_epochs=1,
                      erasing=0,
                      mosaic=0,
                      hsv_h=0,
                      hsv_s=0,
                      hsv_v=0,
                      scale=0,
                      translate=0,
                      )

success = model.export(format='onnx')
