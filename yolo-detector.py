# Import necessary libraries
import cv2
import math
import requests
import base64
from dotenv import dotenv_values
from pytube import YouTube
from ultralytics import YOLO
import cvzone
import numpy as np

print('\nSetting up variables...')

# Load environment variables
CONFIG = dotenv_values('.env')
TB_DEVICE_TELTRY_ENDPOINT = CONFIG['TB_DEVICE_TELTRY_ENDPOINT']
TB_DEVICE_ATTS_ENDPOINT = CONFIG['TB_DEVICE_ATTS_ENDPOINT']
RTSP_CAM_URL = CONFIG['RTSP_CAM_URL']

# SOURCES

# YouTube video URL
youtube_url = "https://www.youtube.com/watch?v=HOk8siZLZqk"
# Get video stream URL
youtube = YouTube(youtube_url)
yt_video_stream = youtube.streams.filter(file_extension="mp4").first()

rtsp_cam = RTSP_CAM_URL

video_sources = [yt_video_stream, rtsp_cam]

print('\nConecting stream source...')
# OpenCV video capture
cap = cv2.VideoCapture(video_sources[1])

# Check if the video capture is successful
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara o video.")
    exit()

# Object detection class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Unwanted object detection classes
unwanted_classes = ['chair', 'tvmonitor', 'boat', "clock"]
unwanted_indices = [classNames.index(cls) for cls in unwanted_classes if cls in classNames]

# YOLO model initialization
print('\nLoading YOLO model...')
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Function to send image as base64 to a specified URL
def send_image_as_base64(url, image):
    retval, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    data = {
        'fotogenica': f"data:image/jpeg;base64,{image_base64}"
    }
    response = requests.post(url, json=data)
    return response

# Initial object detection counts
last_object_counts = {}

print('\nInitializing reading loop...')
# Main loop for video processing
while True:
    # Read a frame from the video stream
    success, img = cap.read()

    # Check if the frame is successfully read
    if not success:
        print("Error: No se pudo leer el frame.")
        break

    # Perform object detection using YOLO model
    results = model(img, verbose=False)[0] #save=True, 

    # Dictionary to store current object counts
    current_object_counts = {}

    print('.',end=' ')
    # Loop through the detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            bbox = x1, y1, w, h
            cvzone.cornerRect(img, bbox)
            conf = math.ceil(box.conf[0] * 100) / 100
            objClass = int(box.cls[0])

            # Check if the detected class is not unwanted
            if objClass not in unwanted_indices:
                cvzone.putTextRect(img, f'{classNames[objClass]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=2)

                # Update the count of detected objects
                if classNames[objClass] in current_object_counts:
                    current_object_counts[classNames[objClass]] += 1
                else:
                    current_object_counts[classNames[objClass]] = 1
                    
    # Process only when more than 1 object and no repeated
    if len(current_object_counts) != len(last_object_counts) and len(current_object_counts) > 1:
        
        
        print("Cambios detectados en los objetos.")
        print("Payload to be post:", current_object_counts)

        # Send the counts to the specified endpoint
        response = requests.post(TB_DEVICE_TELTRY_ENDPOINT, json=current_object_counts)
        print(f"Teltry response Code: {response.status_code}")
        print(f"Teltry response Content: {response.text}")

        # Send the image where the change occurred
        image_response_test = send_image_as_base64(TB_DEVICE_ATTS_ENDPOINT, img)
        print(f"Atts img response Code: {image_response_test.status_code}")
        print(image_response_test.text)


        # cv2.imshow("Video en tiempo real", img)

    # Update the last object counts for comparison in the next iteration
    last_object_counts = current_object_counts

    # Check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()