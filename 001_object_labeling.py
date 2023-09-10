# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:22:07 2023

@author: Bharath
@credits: https://chat.openai.com/
"""

import cv2
import json
from datetime import datetime  # Import datetime for generating timestamps

class ObjectLabeler:
    def __init__(self, video_file_name, output_folder, output_file):
        # Initialize the ObjectLabeler instance with provided inputs
        self.video_file_name = video_file_name
        self.output_folder = output_folder
        self.output_file = output_file
        self.labeled_objects = []

    def label_objects(self):
        try:
            video = cv2.VideoCapture(self.video_file_name)
            current_object = None
            frame_count = 0
            print("Finish the selection process by pressing ESC button!")
            print("Select a ROI and then press SPACE or ENTER button!")
            print("Cancel the selection process by pressing c button!")
            while True:
                success = video.grab()
                if not success:
                    break

                # Read every 10th frame in the video
                if frame_count % 10 == 0:
                    success, frame = video.retrieve()
                    if not success:
                        break

                    # Select regions of interest (ROIs) in the frame
                    rois = cv2.selectROIs("Label", frame, False)
                    for roi in rois:
                        (x, y, w, h) = tuple(map(int, roi))
                        if w == 0 or h == 0:
                            continue
                        # Store labeled object's coordinates and frame ID
                        current_object = {"x": x, "y": y, "w": w, "h": h, "frame_id": frame_count}
                        self.labeled_objects.append(current_object)
                        current_object = None
                        # Draw bounding boxes on the frame
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Save the frame with bounding boxes
                    cv2.imwrite(f"{self.output_folder}frame_{frame_count}.jpg", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

                frame_count += 1

            # Release the video and close windows
            video.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(e)

    def save_labeled_objects(self):
        # Convert the labeled_objects list to a JSON-formatted string
        labeled_objects_json = json.dumps(self.labeled_objects)
        # Write the JSON string to the output file
        with open(self.output_file, 'w') as file:
            file.write(labeled_objects_json)



video_file_name = "Resources/Videos/tyra.mp4"
output_folder = "Resources/SavedLabels/labeled_frames/"

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = f"labeled_objects_{current_time}.json"

# Create an instance of the ObjectLabeler class
labeler = ObjectLabeler(video_file_name, output_folder, output_file)
# Label objects in the video frames
labeler.label_objects()
# Save the labeled objects to a JSON file
labeler.save_labeled_objects()
