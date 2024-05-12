import cv2
import numpy as np
from PIL import Image

# Load the pre-trained Haar cascades for person detection
person_cascade = cv2.CascadeClassifier('D:/Coding/Hackathons/Innocodathon/static/misc/haarcascade_fullbody.xml')

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])*2 + (point1[1] - point2[1])*2)

# Function to detect persons in the frame and draw bounding boxes
def detect_persons(frame,persons):
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
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(2, 2  ))
        frame, centroids = detect_persons(frame,persons)
        stampede = check_stampede(centroids, threshold)

        frame = cv2.resize(frame, (960, 540))
        if stampede:
            for (x, y, w, h) in persons:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2,)
        else:
            for (x, y, w, h) in persons:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2,)

        i += 1
        frames.append(frame)
        cv2.imshow("Frames",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    ret, frame = cap.read()
    height,width = 960, 540
    video=cv2.VideoWriter('video.avi',fourcc,30,(width,height))
    for i in range(30):
        img = cv2.imread("test_store/" + str(i) + '.png')
        video.write(img)
    video.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'D:/Coding/Hackathons/Innocodathon/static/video/crowd.mp4'
    threshold = 4 # Adjust threshold according to your requirement
    new = Image.open("D:\Coding\Hackathons\Innocodathon\static\images\crowd-50.png")
    print(new)