import cv2
import numpy as np
import pygame
import time
import os
from datetime import datetime

# Load YOLOv3-tiny
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

pygame.mixer.init()
alarm_sound = 'alarm.wav'
pygame.mixer.music.load(alarm_sound)

pics_dir = 'pics'
if not os.path.exists(pics_dir):
    os.makedirs(pics_dir)
    print(f"Created directory: {pics_dir}")

detection_counter = 0
detection_threshold = 5  # Number of consecutive frames with detection to trigger alarm
last_image_saved_at = datetime.now()
alarm_playing = False

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

                detection_counter += 1

        if detection_counter >= detection_threshold and not alarm_playing:
            print("Upper body consistently detected, playing alarm...")
            pygame.mixer.music.play()
            alarm_playing = True

        if alarm_playing and not pygame.mixer.music.get_busy():
            alarm_playing = False

        # Reset detection counter if no person is detected
        if len(boxes) == 0:
            detection_counter = 0
            alarm_playing = False

        cv2.imshow('Upper Body Detection', frame)

        current_time = datetime.now()
        time_diff = current_time - last_image_saved_at
        if time_diff.total_seconds() >= 5 and len(boxes) > 0:
            last_image_saved_at = current_time
            filename = os.path.join(pics_dir, f'frame_{current_time.strftime("%Y%m%d%H%M%S")}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break


    except Exception as e:
        print(f"Unexpected error: {e}")

cap.release()
cv2.destroyAllWindows()
print("Video capture released and all OpenCV windows closed.")
