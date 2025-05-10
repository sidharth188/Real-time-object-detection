import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # use yolov8s.pt or better if your system can handle

# Open webcam
cap = cv2.VideoCapture(0)

# Video writer for saving output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('tracked_output.avi',
                      cv2.VideoWriter_fourcc(*'XVID'),
                      20.0,
                      (frame_width, frame_height))

# Setup window
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 800, 600)

# Track unique IDs
unique_ids = set()
prev_time = 0

print("Tracking... Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")

    if results[0].boxes.id is not None:
        for box, cls, track_id in zip(results[0].boxes.xyxy,  # bounding boxes
                                      results[0].boxes.cls,   # class indexes
                                      results[0].boxes.id):   # object IDs

            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]
            id_num = int(track_id)

            unique_ids.add(id_num)

            label = f"{class_name} #{id_num}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Show unique ID count
    cv2.putText(frame, f"Unique Objects: {len(unique_ids)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display and save
    cv2.imshow("Tracking", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
