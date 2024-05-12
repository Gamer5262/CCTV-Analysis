import cv2
import numpy as np
import requests

# Load the pre-trained Haar cascades for person detection
person_cascade = cv2.CascadeClassifier('D:/Coding/Hackathons/Innocodathon/static/misc/haarcascade_fullbody.xml')

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
def main(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
        cv2.imshow("OUTPUT", frame)
        _, imdata = cv2.imencode('.JPG', frame)

        print('.', end='', flush=True)

        requests.put('http://127.0.0.1:5000/upload', data=imdata.tobytes())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'D:/Coding/Hackathons/Innocodathon/static/video/crowd.mp4'
    threshold = 0.5 # Adjust threshold according to your requirement
    main(video_path, threshold)