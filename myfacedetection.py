import cv2
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Print current directory for debugging
print(f"Current directory: {current_dir}")

# Load the cascades
face_cascade_path = os.path.join(current_dir, 'haarcascade_frontalcatface.xml')
eye_cascade_path = os.path.join(current_dir, 'haarcascade_eye.xml')

print(f"Face cascade path: {face_cascade_path}")
print(f"Eye cascade path: {eye_cascade_path}")

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Check if cascades loaded successfully
if face_cascade.empty():
    print("Error: Could not load face cascade")
    exit()
if eye_cascade.empty():
    print("Error: Could not load eye cascade")
    exit()

print("Cascades loaded successfully!")

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

print("Video capture opened successfully!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Region of interest for eyes
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Face and Eye Detection', frame)
    
    # Break loop on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
