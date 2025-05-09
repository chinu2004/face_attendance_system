# face_lock_system.py
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm

# Load your embeddings
my_embeddings = np.load("my_embeddings.npy")

# Initialize
detector = MTCNN()
embedder = FaceNet()
cap = cv2.VideoCapture(0)

# Set strict threshold
THRESHOLD = 0.70 # Lower = more strict

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

        # Compare with all 55 embeddings
        distances = [norm(embedding - my_emb) for my_emb in my_embeddings]
        min_distance = min(distances)

        if min_distance < THRESHOLD:
            label = f"Esther jasmine ({min_distance:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Other ({min_distance:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Lock System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
