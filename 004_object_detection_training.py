# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:50:13 2023

@author: Bharath
@credits: https://chat.openai.com/
"""

# Object Detection Training
import cv2
import json
import os
import numpy as np
import time  # Import the time module for timestamp
from datetime import datetime  # Import datetime for generating timestamps
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torch.utils.data import DataLoader
import pickle
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

current_directory = os.getcwd().replace("\\", "/")
json_file_name = "Resources/SavedObjects/color_bounds.json"
json_file_path = current_directory + "/" + json_file_name
# Load the bounds from the JSON file
with open(json_file_path, "r") as json_file:
    bounds_data = json.load(json_file)

# Convert the loaded lists back to NumPy arrays
lower_bound = np.array(bounds_data["lower_bound"])
upper_bound = np.array(bounds_data["upper_bound"])

class ObjectDetectionDataset(Dataset):
    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations
        

    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def extract_color_component(image, lower_bound, upper_bound):
        color_mask = cv2.inRange(image, lower_bound, upper_bound)
        return cv2.bitwise_and(image, image, mask=color_mask)

    def __getitem__(self, index):
        image = self.images[index]
        annotation = self.annotations[index]


        # Extract the bounding box coordinates from the annotation
        x, y, xx, yy = annotation["x_min"], annotation["y_min"], annotation["x_max"], annotation["y_max"]

        # Create the target tensor containing the bounding box coordinates
        # Check if the bounding box is valid
        if xx > x and yy > y:
            # Create the target tensor containing the bounding box coordinates
            target = {
                "boxes": torch.tensor([[x, y, xx, yy]], dtype=torch.float32),
                "labels": torch.tensor([annotation["class"]], dtype=torch.int64),
            }
        else:
            # Invalid bounding box, return None as the target
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64)
            }


        color_component = self.extract_color_component(image, lower_bound, upper_bound)

        color_tensor = F.to_tensor(color_component)
        # Convert the extracted color to a tensor
        color_tensor = torch.tensor(color_tensor, dtype=torch.float32)

        # Return both the image tensor and color tensor as features
        return color_tensor, target


class ObjectDetectionTrainer:
    def __init__(self, dataset, model, device='cuda'):
        self.dataset = dataset
        self.model = model
        self.device = torch.device(device)

        # Define the optimizer and learning rate scheduler
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def train(self, batch_size=4, num_epochs=10):
        # Create a data loader for the dataset
        print("Calling DataLoader")
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

        for epoch in range(num_epochs):
            print(f"Starting epoch number {epoch}")
            start_time = time.time()  # Start the timer for the current epoch
            self.model.train()

            for images, targets in data_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Forward pass
                outputs = self.model(images, targets)

                # Compute the loss
                loss_classifier = outputs["loss_classifier"]
                loss_box_reg = outputs["loss_box_reg"]
                loss_objectness = outputs["loss_objectness"]
                loss_rpn_box_reg = outputs["loss_rpn_box_reg"]

                # Compute the total loss
                losses = sum(loss for loss in outputs.values())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            # Update the learning rate
            self.lr_scheduler.step()

            end_time = time.time()  # Stop the timer for the current epoch
            epoch_time = end_time - start_time

            # Print the training loss for this epoch
            print(f"Epoch {epoch+1}/{num_epochs}, loss_classifier: {loss_classifier.item()}, Time: {epoch_time:.2f} seconds")
            print(f"Epoch {epoch+1}/{num_epochs}, loss_box_reg: {loss_box_reg}")
            print(f"Epoch {epoch+1}/{num_epochs}, loss_objectness: {loss_objectness}")
            print(f"Epoch {epoch+1}/{num_epochs}, loss_rpn_box_reg: {loss_rpn_box_reg}")
            # Update the statistics dictionary
            training_stats = {}
            training_stats["epoch"] = (epoch + 1)
            training_stats["loss_classifier"] = (loss_classifier.item())
            training_stats["loss_box_reg"] = (loss_box_reg.item())
            training_stats["loss_objectness"] = (loss_objectness.item())
            training_stats["loss_rpn_box_reg"] = (loss_rpn_box_reg.item())


            # Save the trained model with a unique timestamp in the filename
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_file_name = f"Resources/SavedObjects/faster_rcnn_model_epoch_{epoch+1}_{current_time}.pth"
            model_file_path = current_directory + "/" + model_file_name
            torch.save(self.model.state_dict(), model_file_path)
            
            # Specify the file path where you want to save the statistics
            stats_file_name = f"Resources/SavedObjects/training_stats_{epoch+1}_{current_time}.json"
            stats_file_path = current_directory + "/" + stats_file_name
            # Save the statistics to the file
            with open(stats_file_path, "w") as stats_file:
                json.dump(training_stats, stats_file, indent=4)

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# Preprocessed data file
preprocessed_data_file_name = 'Resources/SavedObjects/object_detection_data_preprocessed_2023-09-08_23-11-14.pkl'
preprocessed_data_file_path = current_directory + "/" + preprocessed_data_file_name

# Load dataset object containing preprocessed images and annotations
print(f"BEGIN: Loading file {preprocessed_data_file_path}")
with open(preprocessed_data_file_path, 'rb') as f:
    data = pickle.load(f)
print(f"END: Loading file {preprocessed_data_file_path}")

dataset = ObjectDetectionDataset(data["images"], data["annotations"])

# Define the model architecture
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Modify the model to match the number of classes in your dataset
num_classes = 2  # Number of classes in your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Set device for training (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", "cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create an object detection trainer
trainer = ObjectDetectionTrainer(dataset, model, device)

# Train the model
trainer.train(batch_size=4, num_epochs=10)
