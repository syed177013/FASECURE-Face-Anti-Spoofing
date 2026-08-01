import os
import cv2
import shutil
import random

# Input and output paths
DATASET_DIR = "C:\\Users\\Admin\\Documents\\CelebA_Spoof\\Data\\train"  # Change if needed
#OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\MEGA\\Fasecure\\dataset\\Celeb_faces_train"
OUTPUT_DIR = "C:\\Users\\Admin\\Documents\\MajorPDatasetFaceExtracted"

TARGET_SIZE = (224, 224)  # Resize extracted faces
SESSION_LIMIT = 1500  # Process only 3000 sessions

# Model files
PROTO_PATH = "C:\\Users\\Admin\\Documents\\MEGA\\MY FILES\\Fasecure\\POSTPURGE\\resources\\deploy.prototxt.txt"
MODEL_PATH = "C:\\Users\\Admin\\Documents\\MEGA\\MY FILES\\Fasecure\\POSTPURGE\\resources\\res10_300x300_ssd_iter_140000.caffemodel"

# Load OpenCV's Deep Learning-based Face Detector
net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

# Ensure output directory is clean
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Get a list of available sessions and randomly select 3000
all_sessions = sorted(os.listdir(DATASET_DIR))  # Sort for consistency
selected_sessions = all_sessions[:SESSION_LIMIT]  # Pick first 3000 sequentially


# Process images
for session in selected_sessions:
    session_path = os.path.join(DATASET_DIR, session)
    
    for label in ["live", "spoof"]:
        input_folder = os.path.join(session_path, label)
        output_folder = os.path.join(OUTPUT_DIR, session, label)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(input_folder):
            print(f"⚠️ Skipped {session}/{label} (Folder not found)")
            continue  # Skip missing folders
        
        for file in os.listdir(input_folder):
            if file.endswith(".jpg"):
                image_path = os.path.join(input_folder, file)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Skipped {file} (Image not found or corrupted)")
                    continue
                
                (h, w) = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                             mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
                
                net.setInput(blob)
                detections = net.forward()

                if detections.shape[2] == 0:
                    print(f"⚠️ Skipped {file} (No face detected)")
                    continue

                # Find the best detection
                best_face = None
                max_confidence = 0.5  # Minimum confidence threshold

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > max_confidence:
                        box = detections[0, 0, i, 3:7] * [w, h, w, h]
                        (x1, y1, x2, y2) = box.astype("int")

                        # Ensure valid bounding box
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        best_face = image[y1:y2, x1:x2]
                        max_confidence = confidence

                if best_face is None or best_face.size == 0:
                    print(f"⚠️ Skipped {file} (Invalid face crop)")
                    continue

                # Resize and save
                face_resized = cv2.resize(best_face, TARGET_SIZE)
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, face_resized)

print(f"✅ Face Extraction Done! Processed {SESSION_LIMIT} sessions. Data saved in:", OUTPUT_DIR)
