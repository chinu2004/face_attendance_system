# save_embeddings.py
import os
import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize
detector = MTCNN()
embedder = FaceNet()

def extract_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        return face
    return None

dataset_path = 'dataset'
embeddings = []
names = []

# Loop through each person folder
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing {person_name}...")

    for filename in os.listdir(person_folder):
        path = os.path.join(person_folder, filename)
        face = extract_face(path)
        if face is not None:
            embedding = embedder.embeddings([face])[0]
            embeddings.append(embedding)
            names.append(person_name)

# Save embeddings
embeddings = np.array(embeddings)
np.save("my_embeddings.npy", embeddings)

# Save names
df = pd.DataFrame(names, columns=["name"])
df.to_csv("students.csv", index=False)

print(f"✅ Saved {len(names)} face embeddings to my_embeddings.npy")
print(f"✅ Saved labels to students.csv ({len(set(names))} persons)")
