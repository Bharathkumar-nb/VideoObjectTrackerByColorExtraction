# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 01:05:34 2023

@author: Bharath
"""

import cv2
import json
import numpy as np
import os


current_directory = os.getcwd().replace("\\", "/")
video_file_name = "Resources/Videos/tyra.mp4"
video_file = current_directory + "/" + video_file_name

# Define the file path where you want to save the bounds
json_file_name = "Resources/SavedLabels/color_bounds.json"
json_file_path = current_directory + "/" + json_file_name

frames = [2010, 2590, 6720, 6790, 7780]

def grab_nth_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    # Set the frame position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    # Read and store the nth frame
    success, nth_frame = cap.read()

    # Release the video capture object
    cap.release()

    if success:
        return nth_frame
    else:
        print("Error: Could not read the nth frame.")
        return None


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the color of the clicked pixel
        color = image[y, x]
        sampled_colors.append(color)


# Global variables to store sampled colors
sampled_colors = []

for frame_number in frames:
    # Load the image
    image = grab_nth_frame(video_file_name, frame_number)

    # Create a window to display the image
    cv2.imshow("Sample Colors", image)

    # Set the mouse callback function
    cv2.setMouseCallback("Sample Colors", mouse_callback)

    # Instructions for the user
    print("Click on pixels to sample colors. Press 'q' when finished.")

    # Continuously update the image display until the user presses 'q'
    while True:
        cv2.imshow("Sample Colors", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Cleanup
cv2.destroyAllWindows()

# Print or use the sampled colors for further processing
print(f"Sampled colors ({len(sampled_colors)}): {sampled_colors}\n\n")

# Convert sampled_colors to a NumPy array for easier calculations
sampled_colors_array = np.array(sampled_colors)

# Calculate the mean and standard deviation of the sampled colors
mean_color = np.mean(sampled_colors_array, axis=0)
std_dev_color = np.std(sampled_colors_array, axis=0)

# Define a threshold to determine outliers (e.g., 2 standard deviations)
threshold = 1

# Filter out colors that are outliers
filtered_colors = [color for color in sampled_colors if np.all(np.abs(np.array(color) - mean_color) < threshold * std_dev_color)]

# Now, filtered_colors contains the sampled colors with outliers removed
print(f"filtered_colors ({len(filtered_colors)}): {filtered_colors}")

# Calculate the minimum and maximum values for each color channel
min_b = min(color[0] for color in filtered_colors)
min_g = min(color[1] for color in filtered_colors)
min_r = min(color[2] for color in filtered_colors)

max_b = max(color[0] for color in filtered_colors)
max_g = max(color[1] for color in filtered_colors)
max_r = max(color[2] for color in filtered_colors)

# Calculate the single lower and upper bounds with tolerance
color_tolerance = 15  # Adjust the tolerance as needed
lower_bound = np.array([max(0, min_b - color_tolerance),
                        max(0, min_g - color_tolerance),
                        max(0, min_r - color_tolerance)], dtype=np.uint8)

upper_bound = np.array([min(255, max_b + color_tolerance),
                        min(255, max_g + color_tolerance),
                        min(255, max_r + color_tolerance)], dtype=np.uint8)

# Now, lower_bound and upper_bound represent the color range based on sampled colors

print(lower_bound, upper_bound)

def extract_and_show_colored_video(video_file, lower_bound, upper_bound):
    # Open the video capture
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Video file not found or could not be opened.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Create a mask for the specific color with dtype=np.uint8
        color_mask = cv2.inRange(frame, lower_bound, upper_bound)

        # Apply the mask to the original frame to extract the specific color(s)
        result_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
        
        # Display the resulting frame
        cv2.imshow("Actual Video", frame)
        
        # Display the resulting frame
        cv2.imshow("Extracted Colors Video", result_frame)

        # Exit the loop when the user presses the 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

extract_and_show_colored_video(video_file, lower_bound, upper_bound)

# Convert the NumPy arrays to Python lists
filtered_colors_list = [color.tolist() for color in filtered_colors]

# Create a dictionary to store the bounds
bounds_data = {
    "filtered_colors": filtered_colors_list,
    "threshold": threshold,
    "color_tolerance": color_tolerance,
    "lower_bound": lower_bound,
    "upper_bound": upper_bound
}


# Serialize and save the bounds to a JSON file
with open(json_file_path, "w") as json_file:
    json.dump(bounds_data, json_file)