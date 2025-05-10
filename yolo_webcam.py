from ultralytics import YOLO
import cv2

# Load a YOLOv8 model (you can also use yolov8n.pt, yolov8s.pt, etc.)
model = YOLO("yolov8n.pt")  # 'n' is the nano version – fastest for webcam

# Open the webcam (0 is default camera)
cap = cv2.VideoCapture(0)

# Loop through the video frames
print("Press 'Q' or 'ESC' to quit the webcam.")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, show=True)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'): 
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
