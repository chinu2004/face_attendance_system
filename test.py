from flask import Flask, render_template, Response, send_file, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from datetime import datetime
from threading import Thread, Lock

app = Flask(__name__)

# Load the model and labels
model = tf.keras.models.load_model("face_classifier.keras")
class_labels = ['Seetha', 'Elcy Gold', 'Chinu']
confidence_threshold = 0.85
img_size = 224

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# CSV file path
csv_file = "students.csv"

# Function to initialize the CSV with all students marked as Absent
def initialize_csv():
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Status", "Start Time"])
        for name in class_labels:
            writer.writerow([name, "Absent", ""])

# Initialize variables
capture = False
attendance_log = set()
lock = Lock()

# Generator function to process video frames
def generate_frames():
    global capture, attendance_log
    cap = cv2.VideoCapture(0)

    while capture:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (img_size, img_size))
            face_normalized = face_resized.astype('float32') / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            preds = model.predict(face_input, verbose=0)
            class_index = np.argmax(preds)
            confidence = np.max(preds)

            if confidence >= confidence_threshold:
                name = class_labels[class_index]
                label = f"{name} ({confidence:.2f})"

                with lock:
                    if name not in attendance_log:
                        attendance_log.add(name)
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        with open(csv_file, mode='r', newline='') as file:
                            rows = list(csv.reader(file))

                        for i, row in enumerate(rows):
                            if row[0] == name:
                                rows[i][1] = "Present"
                                rows[i][2] = current_time
                                break

                        with open(csv_file, mode='w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows(rows)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route to display index page
@app.route('/')
def index():
    initialize_csv()
    return render_template('index.html')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to start the camera
@app.route('/start')
def start():
    global capture
    capture = True
    return "Camera started"

# Route to stop the camera
@app.route('/stop')
def stop():
    global capture
    capture = False
    return "Camera stopped"

# Route to download the CSV file
@app.route('/download')
def download():
    return send_file(csv_file, as_attachment=True)

# Route to display the CSV content
@app.route('/display_csv')
def display_csv():
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    return jsonify(data)

# Route to restart the system (reset the CSV)
@app.route('/restart')
def restart_system():
    initialize_csv()
    global attendance_log
    attendance_log = set()
    return "System restarted, all students are marked as Absent."


if __name__ == '__main__':
    app.run(debug=True)
