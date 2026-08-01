def main():

    import cv2
    import torch
    import numpy as np
    from torchvision import transforms
    from src.cdcn_module import CDCNPlusPlusClassifier

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CDCNPlusPlusClassifier()

    model.load_state_dict(
        torch.load(
            "models/fasecure_print_replay.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    net = cv2.dnn.readNetFromCaffe(
        "resources/deploy.prototxt.txt",
        "resources/res10_300x300_ssd_iter_140000.caffemodel"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                input_tensor = preprocess(face).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    label = "Spoof" if prob > 0.5 else "Live"
                    color = (0, 0, 255) if label == "Spoof" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}: {prob:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if not face_detected:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("FAS - Real Time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()