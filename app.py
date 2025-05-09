from flask import Flask, render_template, Response, send_file, jsonify
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
from threading import Lock

app = Flask(__name__)

# Load embeddings
my_embeddings = np.load("my_embeddings.npy")

# Setup
detector = MTCNN()
embedder = FaceNet()
capture = False
lock = Lock()
attendance_logged = False
THRESHOLD = 0.70
csv_file = "students.csv"
username = "Esther Jasmine"

# CSV init
def initialize_csv():
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Status", "Time"])
        writer.writerow([username, "Absent", ""])

# Video processing
def generate_frames():
    global capture, attendance_logged
    cap = cv2.VideoCapture(0)

    while capture:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(rgb)

        for res in results:
            x, y, w, h = res['box']
            face = rgb[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (160, 160))
            except:
                continue

            embedding = embedder.embeddings([face])[0]
            distances = [norm(embedding - e) for e in my_embeddings]
            min_distance = min(distances)

            if min_distance < THRESHOLD:
                label = f"Esther Jasmine"
                color = (0, 255, 0)

                with lock:
                    if not attendance_logged:
                        attendance_logged = True
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(csv_file, 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(["Name", "Status", "Time"])
                            writer.writerow([username, "Present", now])

            else:
                label = f"Others"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def index():
    initialize_csv()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global capture
    capture = True
    return "Camera started"

@app.route('/stop')
def stop():
    global capture
    capture = False
    return "Camera stopped"

@app.route('/download')
def download():
    return send_file(csv_file, as_attachment=True)

@app.route('/display_csv')
def display_csv():
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        return jsonify(list(reader))

@app.route('/restart')
def restart_system():
    global attendance_logged
    attendance_logged = False
    initialize_csv()
    return "System restarted. Status: Absent"

if __name__ == '__main__':
    app.run(debug=True)
