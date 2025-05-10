import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import sqlite3

# Load YOLOv8s model
model = YOLO('yolov8s.pt')

# Open webcam or video
cap = cv2.VideoCapture(0)  # use "input_video.mp4" for video file

# Frame info
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Output video
out = cv2.VideoWriter('tracked_output.avi',
                      cv2.VideoWriter_fourcc(*'XVID'),
                      20.0,
                      (frame_width, frame_height))

# CSV log
log_file = "object_log.csv"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("timestamp,object_count\n")

# SQLite DB
conn = sqlite3.connect("detections.db")
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS detections (
    timestamp TEXT,
    frame_id INTEGER,
    object TEXT,
    confidence REAL,
    x REAL,
    y REAL,
    width REAL,
    height REAL
)''')
conn.commit()

# Fullscreen
cv2.namedWindow("YOLOv8 Tracking", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLOv8 Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_id = 0
print("Tracking... Press 'Q' or 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", stream=False)
    annotated_frame = results[0].plot()

    detections = results[0].boxes
    object_count = 0
    timestamp = datetime.now()

    # Log to CSV + SQLite
    for box in detections:
        cls_id = int(box.cls.cpu().numpy()[0])
        label = model.names[cls_id]
        conf = float(box.conf.cpu().numpy()[0])
        x, y, w, h = box.xywh[0].tolist()

        object_count += 1

        cursor.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (timestamp, frame_id, label, conf, x, y, w, h))

    conn.commit()

    # Write to CSV
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{object_count}\n")

    # Show object count
    cv2.putText(annotated_frame,
                f'Object Count: {object_count}',
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
                cv2.LINE_AA)

    # Show and save
    cv2.imshow("YOLOv8 Tracking", annotated_frame)
    out.write(annotated_frame)

    frame_id += 1
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

cap.release()
out.release()
conn.close()
cv2.destroyAllWindows()

# Run EDA script after YOLO exits 
import subprocess
subprocess.run(["python", "eda.py"])
