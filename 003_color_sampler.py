# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:05:34 2023

@author: Bharath
"""

import cv2
import json
import numpy as np
import os



class ColorSampler:
    def __init__(self, video_file_name, frames, json_file_name):
        self.video_file_name = video_file_name
        self.frames = frames
        self.json_file_name = json_file_name
        self.sampled_colors = []
        self.mean_color = None
        self.std_dev_color = None
        self.threshold = 1
        self.filtered_colors = []
        self.lower_bound = None
        self.upper_bound = None
        self.color_tolerance = 15
        self.first = True

    def grab_nth_frame(self, video_path, frame_number):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, nth_frame = cap.read()
        cap.release()
        if success:
            return nth_frame
        else:
            print("Error: Could not read the nth frame.")
            return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            color = self.image[y, x]
            self.sampled_colors.append(color)

    def sample_colors(self):
        for frame_number in self.frames:
            self.image = self.grab_nth_frame(self.video_file_name, frame_number)
            cv2.imshow("Sample Colors", self.image)
            cv2.setMouseCallback("Sample Colors", self.mouse_callback)
            print("Click on pixels to sample colors. Press 'q' when finished.")
            while True:
                cv2.imshow("Sample Colors", self.image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        cv2.destroyAllWindows()

    def process_sampled_colors(self):
        self.sampled_colors_array = np.array(self.sampled_colors)
        self.mean_color = np.mean(self.sampled_colors_array, axis=0)
        self.std_dev_color = np.std(self.sampled_colors_array, axis=0)
        self.filtered_colors = [color for color in self.sampled_colors if np.all(np.abs(np.array(color) - self.mean_color) < self.threshold * self.std_dev_color)]

    def calculate_color_bounds(self):
        min_b = min(color[0] for color in self.filtered_colors)
        min_g = min(color[1] for color in self.filtered_colors)
        min_r = min(color[2] for color in self.filtered_colors)
        max_b = max(color[0] for color in self.filtered_colors)
        max_g = max(color[1] for color in self.filtered_colors)
        max_r = max(color[2] for color in self.filtered_colors)
        self.lower_bound = np.array([max(0, min_b - self.color_tolerance),
                                     max(0, min_g - self.color_tolerance),
                                     max(0, min_r - self.color_tolerance)], dtype=np.uint8)
        self.upper_bound = np.array([min(255, max_b + self.color_tolerance),
                                     min(255, max_g + self.color_tolerance),
                                     min(255, max_r + self.color_tolerance)], dtype=np.uint8)

    def extract_and_show_colored_video(self, video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error: Video file not found or could not be opened.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            color_mask = cv2.inRange(frame, self.lower_bound, self.upper_bound)
            result_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
            
            non_zero_indices = np.nonzero(result_frame)
            # Get the first 5 non-zero indices
            non_zero_indices = (non_zero_indices[0][:5], non_zero_indices[1][:5])

            # if self.first:
            #     # Print the RGB values of the first 5 non-zero pixels
            #     for y, x in zip(*non_zero_indices):
            #         rgb_value = result_frame[y, x]
            #         print(f"result_frame: RGB value at ({x}, {y}): {rgb_value}")
            #     for y, x in zip(*non_zero_indices):
            #         rgb_value = frame[y, x]
            #         print(f"frame: RGB value at ({x}, {y}): {rgb_value}")
            #     self.first = False
            cv2.imshow("Actual Video", frame)
            cv2.imshow("Extracted Colors Video", result_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def save_color_bounds(self):
        # Convert the NumPy arrays to Python lists
        filtered_colors_list = [color.tolist() for color in self.filtered_colors]
        bounds_data = {
            "filtered_colors": filtered_colors_list,
            "threshold": self.threshold,
            "color_tolerance": self.color_tolerance,
            "lower_bound": self.lower_bound.tolist(),
            "upper_bound": self.upper_bound.tolist()
        }
        with open(self.json_file_name, "w") as json_file:
            json.dump(bounds_data, json_file)

    def run(self):
        self.sample_colors()
        self.process_sampled_colors()
        self.calculate_color_bounds()
        self.extract_and_show_colored_video(self.video_file_name)
        #self.save_color_bounds()

if __name__ == "__main__":
    current_directory = os.getcwd().replace("\\", "/")
    video_file_name = "Resources/Videos/tyra.mp4"
    frames = [2010, 2590, 6720, 6790, 7780]
    json_file_name = "Resources/SavedObjects/color_bounds.json"
    json_file_path = current_directory + "/" + json_file_name

    color_sampler = ColorSampler(video_file_name, frames, json_file_path)
    color_sampler.run()
