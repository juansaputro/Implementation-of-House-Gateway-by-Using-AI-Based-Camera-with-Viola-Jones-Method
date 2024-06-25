import cv2
import time
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to check if two faces are the same (based on coordinates)
def same_face(face1, face2):
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    return abs(x1 - x2) < 10 and abs(y1 - y2) < 10 and abs(w1 - w2) < 10 and abs(h1 - h2) < 10

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Dictionary to track detected faces and their detection start time
faces_dict = {}

# Variables for metrics
total_frames = 0
total_actual_faces = 0
total_detected_faces = 0
total_correct_duration = 0

# Variables for additional metrics
false_positives = 0
false_negatives = 0
processing_times = []

start_time = time.time()  # Start time for FPS calculation
program_start_time = time.time()  # Start time for the program

while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break
    
    total_frames += 1
    start_frame_time = time.time()
    
    faces = detect_faces(frame)
    current_time = time.time()
    
    # Update face dictionary
    face_tuples = [tuple(face) for face in faces]
    for face in face_tuples:
        found = False
        for existing_face in list(faces_dict.keys()):
            if same_face(existing_face, face):
                faces_dict[face] = faces_dict.pop(existing_face)
                found = True
                break
        if not found:
            faces_dict[face] = current_time
    
    # Calculate metrics
    total_actual_faces += len(face_tuples)
    total_detected_faces += len(faces_dict)
    
    for face, start_time in list(faces_dict.items()):
        x, y, w, h = face
        if face not in face_tuples:
            del faces_dict[face]
            false_negatives += 1
            continue
        duration = current_time - start_time
        total_correct_duration += 1 if duration >= 3 else 0
        
        # Draw rectangles and labels
        if duration >= 3:
            color = (0, 0, 255)  # Red
        else:
            color = (255, 0, 0)  # Blue
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  # Draw a rectangle around each face
        cv2.putText(frame, 'Manusia', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Add label above the rectangle
    
    # Measure processing time for the current frame
    processing_time = time.time() - start_frame_time
    processing_times.append(processing_time)
    
    cv2.imshow('Face Detection', frame)  # Display the frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate final metrics
total_time = time.time() - start_time  # Total time for FPS calculation
program_run_time = time.time() - program_start_time  # Total time the program has been running
fps = total_frames / total_time  # Frames per second

if total_actual_faces > 0:
    detection_rate = (total_detected_faces / total_actual_faces) * 100
else:
    detection_rate = 0  # Default to 100% if no faces were expected

if total_detected_faces > 0:
    accuracy_duration = (total_correct_duration / total_detected_faces) * 100
    average_response_time = total_correct_duration / total_detected_faces
else:
    accuracy_duration = 0.0
    average_response_time = 0.0

if total_frames > 0:
    average_processing_time = sum(processing_times) / total_frames
else:
    average_processing_time = 0.0

# Assuming we have a method to count true positives
true_positives = total_actual_faces - false_negatives
false_positives = total_detected_faces - true_positives

precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0

print(f"Detection rate: {detection_rate:.2f}%")
print(f"Accuracy of detection duration (>3 seconds): {accuracy_duration:.2f}%")
print(f"FPS: {fps:.2f}")
print(f"Average response time per face: {average_response_time:.2f} seconds")
print(f"Average processing time per frame: {average_processing_time:.4f} seconds")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1-Score: {f1_score:.2f}%")
print(f"The program has been running for: {program_run_time:.2f} seconds")

# Plot the processing times per frame

# Data for plotting
metrics = ['Precision', 'Recall', 'F1-Score']
values = [precision, recall, f1_score]

# Creating the bar chart
plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=['blue', 'green', 'red'])

# Adding the values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.01, f"{yval:.2f}")

# Adding titles and labels
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Face Detection Evaluation Metrics')
plt.ylim(0, 100)

plt.figure(figsize=(10, 5))
plt.plot(processing_times, label='Processing Time per Frame')
plt.xlabel('Frame')
plt.ylabel('Time (seconds)')
plt.title('Processing Time per Frame')
plt.legend()
plt.grid(True)
plt.show()

cap.release()
cv2.destroyAllWindows()