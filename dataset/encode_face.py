# save_embeddings.py
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

# Initialize
detector = MTCNN()
embedder = FaceNet()

def extract_face(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    if faces:
        x, y, w, h = faces[0]['box']
        face = img_rgb[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        return face
    return None

folder = 'dataset/train'
embeddings = []

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    face = extract_face(path)
    if face is not None:
        embedding = embedder.embeddings([face])[0]
        embeddings.append(embedding)

# Save all embeddings
embeddings = np.array(embeddings)
np.save("my_embeddings.npy", embeddings)
print("✅ Saved 55 face embeddings to my_embeddings.npy")
