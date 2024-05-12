from flask import Flask, render_template, request, Response
import cv2
import pafy
import time
import numpy as np
import requests
import urllib.request
from ultralytics import YOLO
from PIL import Image
import onnx
import torch
import os


app = Flask(__name__)
frame = None   # global variable to keep single JPG, 
               # at start you could assign bytes from empty JPG
video_link = None
case = None
person_cascade = cv2.CascadeClassifier('D:/Coding/Hackathons/CCTV-Analysis/static/misc/haarcascade_fullbody.xml')
safety_model = YOLO("D:/Coding/Hackathons/CCTV-Analysis/static/misc/best.torchscript",task='detect')


def anomaly_detect(cap):
    pass


def monitor_detect(cap):
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,640))
    new_frame = safety_model.predict(frame,save=True)
    new_frame = cv2.imread("runs/detect/predict/image0.jpg")
    return new_frame

def gen():
    global case
    global video_link
    url = video_link
    r = requests.get(url)
    with open('video.mp4', 'wb') as outfile:
        outfile.write(r.content)
    cap = cv2.VideoCapture()
    cap.open('video.mp4')     
    
    frame_width = int(cap.get(3))                                        # Returns the width and height of capture video   
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    #print("Video Reolution: ",(width, height))
    
    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect    
    while True:                                                          # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)          # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        
        if case == 'crowd':
            image = crowd_detect(cap)       # Call the function cvDrawBoxes_object() for colored bounding box per class
        elif case == 'anomaly':
            image = anomaly_detect(cap)        # Call the function cvDrawBoxes_social() for colored bounding box per class
        elif case == 'monitor':
            image = monitor_detect(cap)          # Call the function cvDrawBoxes_fall() for colored bounding box per class
                    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'\r\n' + frame + b'\r\n')
                         # 0.04s = 40ms = 25 frames per second 



# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])*2 + (point1[1] - point2[1])*2)

# Function to detect persons in the frame and draw bounding boxes
def detect_persons(frame, persons):
    centroids = []
    for (x, y, w, h) in persons:
        centroids.append((x + w // 2, y + h // 2))
    return frame, centroids

# Function to check for stampede based on distance between centroids
def check_stampede(centroids, threshold):
    for i in range(len(centroids)):
        for j in range(i+1, len(centroids)):
            distance = calculate_distance(centroids[i], centroids[j])
            if distance < threshold:
                return True
    return False

# Main function to process the video
def crowd_detect(cap):
    threshold = 1
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (960, 540))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        frame, centroids = detect_persons(frame, persons)
        stampede = check_stampede(centroids, threshold)

        for (x, y, w, h) in persons:
            if stampede:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Here you can add code to trigger alerts or take further actions
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        return frame
    else:
        return 



@app.route('/video',methods=['GET', 'POST'])
def video():
    global video_link
    video_link = request.form.get('videolink')
    return render_template('Video.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def home():
    return render_template("home.html")
    
@app.route("/crowdmanagement")
def crowdmanagement():
    global case
    case = "crowd"
    return render_template("crowd.html")

@app.route("/abnormalitydetection")
def abnormality():
    global case
    case = "anomaly"
    return render_template("abnormal.html")

@app.route("/workmonitoring")
def monitoring():
    global case
    case = "monitor"
    return render_template("monitor.html")
    
if __name__ == "__main__":
    app.run(debug=True)