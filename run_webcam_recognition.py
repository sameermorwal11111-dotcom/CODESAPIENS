"""
Simple Webcam Face Recognition Script
=====================================
Run this script to start face recognition with your webcam
Press 'Q' to quit
"""

import cv2
from time import time
from tkinter import messagebox
import os

def list_available_users():
    """List all available trained classifiers"""
    classifier_path = "./data/classifiers"
    if not os.path.exists(classifier_path):
        print("No classifiers found. Please train a model first using app-gui.py")
        return []
    
    users = []
    for file in os.listdir(classifier_path):
        if file.endswith("_classifier.xml"):
            user_name = file.replace("_classifier.xml", "")
            users.append(user_name)
    
    return users

def recognize_face(name, timeout=10):
    """
    Recognize faces using trained classifier
    
    Args:
        name: Username to recognize
        timeout: Seconds to run recognition (default 10 seconds)
    """
    print(f"\n{'='*50}")
    print(f"Starting Face Recognition for: {name.upper()}")
    print(f"{'='*50}")
    print("Press 'Q' or wait for timeout to exit")
    print(f"{'='*50}\n")
    
    # Load cascade classifier
    face_cascade_path = './data/haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        print(f"ERROR: Cascade classifier not found at {face_cascade_path}")
        return False
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Load trained recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    classifier_path = f"./data/classifiers/{name}_classifier.xml"
    
    if not os.path.exists(classifier_path):
        print(f"ERROR: Classifier not found for user '{name}'")
        print(f"Expected path: {classifier_path}")
        return False
    
    recognizer.read(classifier_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return False
    
    print("Webcam opened successfully. Loading frames...\n")
    
    pred = False
    start_time = time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            try:
                id, confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                
                # If confidence > 50%, it's a recognized face
                if confidence > 50:
                    pred = True
                    text = f'RECOGNIZED: {name.upper()} ({confidence}%)'
                    color = (0, 255, 0)  # Green for recognized
                else:
                    pred = False
                    text = f"Unknown Face ({confidence}%)"
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, color, 2, cv2.LINE_AA)
                
            except Exception as e:
                print(f"Error during recognition: {e}")
                pred = False
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Error", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.9, (0, 0, 255), 2)
        
        # Display elapsed time
        elapsed_time = time() - start_time
        remaining_time = max(0, timeout - elapsed_time)
        
        # Add info text
        info_text = f"Time: {elapsed_time:.1f}s | Frames: {frame_count} | Status: {'MATCHED' if pred else 'NO MATCH'}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow(f"Face Recognition - {name.upper()}", frame)
        
        # Check for timeout
        if elapsed_time >= timeout:
            print(f"\nTimeout reached ({timeout}s)")
            break
        
        # Check for 'Q' key press
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("\nUser pressed 'Q' - Exiting")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print results
    print(f"\n{'='*50}")
    print("Recognition Complete!")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Result: {'RECOGNIZED' if pred else 'NOT RECOGNIZED'}")
    print(f"{'='*50}\n")
    
    return pred

def main():
    """Main function"""
    print("\n")
    print("╔" + "="*48 + "╗")
    print("║" + " "*10 + "FACE RECOGNITION - WEBCAM" + " "*12 + "║")
    print("╚" + "="*48 + "╝")
    print()
    
    # List available users
    users = list_available_users()
    
    if not users:
        print("❌ No trained classifiers found!")
        print("Please follow these steps:")
        print("1. Run 'python app-gui.py'")
        print("2. Sign up a user")
        print("3. Capture 300+ face images")
        print("4. Train the model")
        print("Then come back and run this script.\n")
        return
    
    print("✓ Available users for recognition:")
    for i, user in enumerate(users, 1):
        print(f"  {i}. {user}")
    print()
    
    # Get user input
    while True:
        choice = input("Enter user number, username, or 'Q' to quit: ").strip()
        
        if choice.upper() == 'Q':
            print("Exiting...\n")
            return
        
        selected_user = None
        
        # Try to match by number
        try:
            index = int(choice) - 1
            if 0 <= index < len(users):
                selected_user = users[index]
        except ValueError:
            pass
        
        # Try to match by username (case-insensitive)
        if not selected_user:
            for user in users:
                if user.lower() == choice.lower():
                    selected_user = user
                    break
        
        if selected_user:
            print(f"\n✓ Selected user: {selected_user}")
            break
        else:
            print("❌ Invalid input. Please enter a number or username from the list above.")
    
    print()
    
    # Run recognition
    recognize_face(selected_user, timeout=10)

if __name__ == "__main__":
    main()
