import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 model (Make sure you have the model file)
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Set up VideoWriter to save the output video
out = cv2.VideoWriter('output.avi',
                      cv2.VideoWriter_fourcc(*'XVID'),
                      20.0,
                      (frame_width, frame_height))

# Set the window to full screen
cv2.namedWindow("Webcam", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Recording... Press 'Q' or 'ESC' to stop.")

# FPS calculation setup
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract detected objects
    detections = results[0].boxes
    object_count = len(detections)

    # Draw the bounding boxes and label the objects
    frame = results[0].plot()

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2, cv2.LINE_AA)

    # Display object count on the frame
    cv2.putText(frame, f"Objects detected: {object_count}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Save the current frame to the video file
    out.write(frame)

    # Break the loop if 'Q' or 'ESC' is pressed
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
