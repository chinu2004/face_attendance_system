# app.py
from flask import Flask, render_template, Response, send_file, jsonify, request
import cv2
import numpy as np
import os
import csv
from datetime import datetime
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
from threading import Lock
import pandas as pd
import time
import uuid

app = Flask(__name__)

# ----------------- CONFIG -----------------
DATASET_PATH = r"C:\Users\USER\Desktop\face attandance system\dataset"
ATTENDANCE_DIR = r"C:\Users\USER\Desktop\face attandance system\attendance_records"
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# Embedding / label files
EMBED_FILE = "my_embeddings.npy"
CSV_FILE = "students.csv"

# Threshold for recognition (adjust experimentally: 0.6-0.9)
THRESHOLD = 0.70

# Globals
capture = False
lock = Lock()                          # for attendance file and shared data
attendance_logged = set()              # names already marked for today (loaded on start)

detector = MTCNN()
embedder = FaceNet()

# ------------- Utility: Attendance CSV -------------
def today_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")

def initialize_today_csv():
    """Ensure today's CSV exists and populate attendance_logged."""
    csv_path = today_filename()
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Status", "Timestamp"])

    global attendance_logged
    with lock:
        attendance_logged = set()
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("Name") or row.get("name") or "").strip()
                    status = (row.get("Status") or row.get("status") or "").strip()
                    if name and status and status.lower().startswith("present"):
                        attendance_logged.add(name)
        except Exception as e:
            print("[WARN] Could not read today's CSV:", e)

# Call on startup
initialize_today_csv()

# ------------- Embeddings load/save helpers -------------
def load_embeddings():
    """Load embeddings and student names. Return (embeddings_array, names_list)."""
    embeddings = np.array([])
    names = []
    if os.path.exists(EMBED_FILE):
        try:
            embeddings = np.load(EMBED_FILE, allow_pickle=False)
            # ensure 2D array when single embedding was saved as 1D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        except Exception as e:
            print("[WARN] Could not load embeddings:", e)
            embeddings = np.array([])
    if os.path.exists(CSV_FILE):
        try:
            names = pd.read_csv(CSV_FILE)["name"].tolist()
        except Exception as e:
            print("[WARN] Could not load student names:", e)
            names = []
    return embeddings, names

# Initialize in-memory embeddings and names
my_embeddings, students = load_embeddings()

# ------------- Update / Append embeddings for a single student -------------
def update_embeddings(new_student):
    """
    Process a single new student's folder and append computed embeddings + name.
    Expects images placed in DATASET_PATH/new_student/*.jpg
    """
    global my_embeddings, students

    student_path = os.path.join(DATASET_PATH, new_student)
    if not os.path.isdir(student_path):
        print("[ERROR] No folder found for", new_student)
        return

    new_embeddings = []
    new_names = []
    for filename in os.listdir(student_path):
        path = os.path.join(student_path, filename)
        # Only process image files
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            faces = detector.detect_faces(rgb)
        except Exception as e:
            print("[WARN] Detector error while processing file", path, e)
            faces = []

        if not faces:
            continue

        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        # crop and resize face
        face = rgb[y:y+h, x:x+w]
        try:
            face = cv2.resize(face, (160, 160))
        except Exception:
            continue
        try:
            emb = embedder.embeddings([face])[0]
        except Exception as e:
            print("[WARN] Embedding error for", path, e)
            continue

        new_embeddings.append(emb)
        new_names.append(new_student)

    if new_embeddings:
        new_embeddings_arr = np.array(new_embeddings)
        # Append to existing embeddings safely
        if my_embeddings.size > 0:
            try:
                my_embeddings = np.vstack([my_embeddings, new_embeddings_arr])
            except Exception:
                # Fallback: re-load embeddings fresh then vstack
                existing, _ = load_embeddings()
                if existing.size > 0:
                    my_embeddings = np.vstack([existing, new_embeddings_arr])
                else:
                    my_embeddings = new_embeddings_arr
        else:
            my_embeddings = new_embeddings_arr

        students.extend(new_names)
        # Save to disk
        try:
            np.save(EMBED_FILE, my_embeddings)
            pd.DataFrame(students, columns=["name"]).to_csv(CSV_FILE, index=False)
            print(f"[INFO] Added {len(new_names)} embeddings for {new_student}")
        except Exception as e:
            print("[ERROR] Saving embeddings/names failed:", e)
    else:
        print("[WARN] No valid faces/embeddings found for", new_student)

# ------------- Video / Recognition -------------
def generate_frames():
    """
    Video generator for MJPEG stream.
    Detect faces -> compute embedding -> compare with stored embeddings -> mark attendance.
    """
    global capture, my_embeddings, students, attendance_logged

    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # small warmup

    while capture:
        ret, frame = cap.read()
        if not ret:
            continue

        # Work on a copy for drawing
        draw_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = detector.detect_faces(rgb)
        except Exception as e:
            print("[WARN] MTCNN error in generate_frames:", e)
            results = []

        for res in results:
            x, y, w, h = res['box']
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (160, 160))
            except Exception:
                continue

            # compute embedding
            try:
                embedding = embedder.embeddings([face])[0]
            except Exception as e:
                print("[WARN] FaceNet embed error:", e)
                continue

            # Default unknown
            name = "Unknown"
            color = (0, 0, 255)

            # Matching only if we have embeddings loaded
            with lock:
                embeddings_copy = my_embeddings.copy() if my_embeddings.size > 0 else np.array([])
                students_copy = students.copy()

            if embeddings_copy.size > 0 and len(students_copy) > 0:
                # Ensure embeddings_copy is 2D
                if embeddings_copy.ndim == 1:
                    embeddings_copy = embeddings_copy.reshape(1, -1)

                # compute distances
                try:
                    distances = np.linalg.norm(embeddings_copy - embedding, axis=1)
                except Exception:
                    # fallback to explicit loop (safe)
                    distances = [norm(embedding - e) for e in embeddings_copy]

                min_index = int(np.argmin(distances))
                min_distance = float(distances[min_index])

                if min_distance < THRESHOLD:
                    name = students_copy[min_index]
                    color = (0, 255, 0)

                    # mark attendance safely (no duplicate for today)
                    with lock:
                        if name not in attendance_logged:
                            attendance_logged.add(name)
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            csv_path = today_filename()
                            try:
                                # ensure file exists
                                if not os.path.exists(csv_path):
                                    with open(csv_path, 'w', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow(["Name", "Status", "Timestamp"])
                                with open(csv_path, 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow([name, "Present", now])
                                print(f"[INFO] {name} marked present at {now}")
                            except Exception as e:
                                print("[ERROR] Writing attendance failed:", e)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

            # draw rectangle + label on BGR frame
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(draw_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # encode frame as JPEG
        ret2, buffer = cv2.imencode('.jpg', draw_frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# -------------- Routes ----------------------
@app.route('/')
def index():
    initialize_today_csv()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global capture
    if not capture:
        capture = True
    return "Camera started"

@app.route('/stop')
def stop():
    global capture
    if capture:
        capture = False
    return "Camera stopped"

# ---------------- Add Student Route ----------------
@app.route('/add_student', methods=['POST'])
def add_student():
    """
    Captures 6 images for a new student, saves in dataset/<name>/, and updates embeddings immediately.
    JSON body can include: {"student_name": "name_here"} else a random name is created.
    """
    global capture

    was_running = capture
    if was_running:
        capture = False
        time.sleep(0.5)

    data = request.get_json(silent=True) or {}
    name = (data.get("student_name") or f"student_{uuid.uuid4().hex[:6]}").strip()

    student_path = os.path.join(DATASET_PATH, name)
    os.makedirs(student_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    start_time = time.time()

    while count < 6:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = detector.detect_faces(rgb)
        except Exception:
            results = []

        if results:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)
            face = frame[y:y+h, x:x+w]
            try:
                face_small = cv2.resize(face, (160, 160))
            except Exception:
                continue
            img_name = os.path.join(student_path, f"{name}_img{count+1}.jpg")
            cv2.imwrite(img_name, face_small)
            count += 1
            cv2.waitKey(400)

        if time.time() - start_time > 20:
            break

    cap.release()
    cv2.destroyAllWindows()

    if was_running:
        time.sleep(0.3)
        capture = True

    # Update embeddings in-process (no subprocess)
    update_embeddings(name)

    return jsonify({"status": "success", "message": f"{name} added with {count} images and embeddings updated."})

@app.route('/display_csv')
def display_csv():
    csv_path = today_filename()
    data = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({
                        "Name": row.get("Name") or row.get("name") or "",
                        "Status": row.get("Status") or row.get("status") or "",
                        "Timestamp": row.get("Timestamp") or row.get("Time") or row.get("time") or ""
                    })
        except Exception as e:
            print("[ERROR] reading CSV:", e)
    return jsonify(data)

@app.route('/download')
def download():
    csv_path = today_filename()
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    else:
        return "No attendance file for today.", 404

@app.route('/restart')
def restart_system():
    """Reset attendance_logged and reinitialize today's CSV."""
    global attendance_logged, my_embeddings, students
    with lock:
        attendance_logged = set()
    initialize_today_csv()
    # reload saved embeddings (if changed externally)
    my_embeddings, students = load_embeddings()
    return "System restarted. Status reset."

# --------------- Run ------------------------
if __name__ == '__main__':
    app.run(debug=True)
