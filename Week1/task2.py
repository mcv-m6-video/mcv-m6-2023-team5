import cv2

# Open video file

cap = cv2.VideoCapture("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi")

# Loop through frames
while True:
    # Read frame
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Display frame
    cv2.imshow("Frame", frame)

    # Check for quit command
    if cv2.waitKey(1) == ord('q'):
        break

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()
