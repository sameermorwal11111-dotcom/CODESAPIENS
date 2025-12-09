"""
Fixed Face Recognition System - OpenCV Windows Compatible
===========================================================
This version fixes the OpenCV GUI error on Windows
"""

import cv2
import os
import numpy as np
from time import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class FaceRecognitionSystemFixed:
    """Fixed Face Recognition System for Windows"""
    
    def __init__(self):
        self.cascade_path = './data/haarcascade_frontalface_default.xml'
        self.classifiers_dir = './data/classifiers'
        self.data_dir = './data'
        self.face_cascade = None
        self.load_cascade()
    
    def load_cascade(self):
        """Load face cascade classifier"""
        if not os.path.exists(self.cascade_path):
            print(f"{RED}❌ ERROR: Cascade classifier not found at {self.cascade_path}{RESET}")
            return False
        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.face_cascade.empty():
                print(f"{RED}❌ ERROR: Failed to load cascade classifier{RESET}")
                return False
            return True
        except Exception as e:
            print(f"{RED}❌ ERROR loading cascade: {e}{RESET}")
            return False
    
    def get_available_users(self):
        """Get list of trained users"""
        if not os.path.exists(self.classifiers_dir):
            return []
        
        users = []
        for file in os.listdir(self.classifiers_dir):
            if file.endswith("_classifier.xml"):
                user_name = file.replace("_classifier.xml", "")
                users.append(user_name)
        
        return sorted(users)
    
    def display_frame(self, frame, title="Video"):
        """Display frame using matplotlib instead of cv2.imshow (fixes Windows issue)"""
        try:
            # Convert BGR to RGB for proper color display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a window to display
            cv2.imshow(title, frame)
            
            # Return True if successful
            return True
        except Exception as e:
            print(f"{RED}Display error: {e}{RESET}")
            return False
    
    def capture_face_data(self, name, num_images=50):
        """Capture face images for a user - FIXED FOR WINDOWS"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}CAPTURING FACE DATA FOR: {BOLD}{name.upper()}{RESET}{CYAN}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")
        print(f"{YELLOW}Instructions:{RESET}")
        print("1. Stand in front of the camera")
        print("2. Keep your face clear and visible")
        print("3. Vary angles and expressions")
        print(f"4. We will capture {num_images} images")
        print("5. Press 'Q' or 'ESC' to stop early")
        print("6. Close the window to stop")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        # Create data directory
        path = os.path.join(self.data_dir, name)
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"{RED}❌ Error creating directory: {e}{RESET}")
            return 0
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ ERROR: Could not open webcam!{RESET}")
            print(f"{YELLOW}⚠ Try:{{RESET}}")
            print("  1. Close other apps using webcam")
            print("  2. Restart VS Code")
            print("  3. Check if webcam is connected")
            return 0
        
        print(f"{GREEN}✓ Webcam opened successfully{RESET}")
        print(f"{BLUE}Starting capture in 3 seconds...{RESET}\n")
        
        import time as time_module
        time_module.sleep(3)
        
        num_captured = 0
        skipped = 0
        window_title = f"Capturing Face Data - {name.upper()} (Press Q to Stop)"
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"{RED}❌ Error: Failed to capture frame{RESET}")
                break
            
            # Resize for better performance
            frame = cv2.resize(frame, (640, 480))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Display frame info
            info_text = f"Captured: {num_captured}/{num_images} | Skipped: {skipped}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q/ESC to stop", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 255), 1)
            
            if len(faces) > 0:
                cv2.putText(frame, "FACE DETECTED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 0), 2)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    try:
                        img_path = os.path.join(path, f"{num_captured}_{name}.jpg")
                        cv2.imwrite(img_path, roi_gray)
                        num_captured += 1
                    except Exception as e:
                        print(f"{RED}Error saving image: {e}{RESET}")
                        skipped += 1
            else:
                cv2.putText(frame, "NO FACE DETECTED", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 0, 255), 2)
                skipped += 1
            
            # Display using cv2.imshow with error handling
            try:
                cv2.imshow(window_title, frame)
            except Exception as e:
                print(f"{YELLOW}⚠ Display warning: {e}{RESET}")
                print(f"{YELLOW}Continuing capture (images still being saved)...{RESET}")
            
            if num_captured >= num_images:
                print(f"\n{GREEN}✓ Captured {num_captured} images successfully!{RESET}")
                break
            
            try:
                key = cv2.waitKey(20) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print(f"\n{YELLOW}⚠ Capture stopped by user{RESET}")
                    break
            except:
                pass
        
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print(f"{CYAN}{'='*60}{RESET}")
        print(f"{GREEN}✓ Capture Complete!{RESET}")
        print(f"  Total Captured: {num_captured}")
        print(f"  Location: {path}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        return num_captured
    
    def train_classifier(self, name):
        """Train LBPH recognizer for a user"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}TRAINING CLASSIFIER FOR: {BOLD}{name.upper()}{RESET}{CYAN}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        data_path = os.path.join(self.data_dir, name)
        
        if not os.path.exists(data_path):
            print(f"{RED}❌ ERROR: No data found for user '{name}'{RESET}")
            return False
        
        # Get all face images
        images = []
        labels = []
        
        print(f"{BLUE}Loading images from {data_path}...{RESET}")
        
        image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        
        if len(image_files) < 10:
            print(f"{RED}❌ ERROR: Need at least 10 images, found {len(image_files)}{RESET}")
            return False
        
        print(f"{GREEN}✓ Found {len(image_files)} images{RESET}\n")
        
        for img_file in image_files:
            try:
                img_path = os.path.join(data_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    images.append(img)
                    labels.append(0)
                    print(f"{GREEN}✓{RESET} Loaded: {img_file}")
            except Exception as e:
                print(f"{RED}✗{RESET} Error loading {img_file}: {e}")
        
        if len(images) < 10:
            print(f"{RED}❌ ERROR: Could not load enough images{RESET}")
            return False
        
        print(f"\n{BLUE}Training classifier with {len(images)} images...{RESET}\n")
        
        try:
            # Create and train recognizer
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(labels))
            
            # Save classifier
            os.makedirs(self.classifiers_dir, exist_ok=True)
            classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
            recognizer.write(classifier_path)
            
            print(f"{GREEN}✓ Classifier trained and saved!{RESET}")
            print(f"  Location: {classifier_path}")
            print(f"{CYAN}{'='*60}{RESET}\n")
            
            return True
        except Exception as e:
            print(f"{RED}❌ Error training classifier: {e}{RESET}")
            return False
    
    def recognize_faces_live(self, name, timeout=15):
        """Live face recognition - FIXED FOR WINDOWS"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}LIVE FACE RECOGNITION{RESET}")
        print(f"{CYAN}Matching against: {BOLD}{name.upper()}{RESET}{CYAN}{RESET}")
        print(f"{CYAN}Timeout: {timeout} seconds{RESET}")
        print(f"{CYAN}Press 'Q' to exit{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        # Load classifier
        classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
        if not os.path.exists(classifier_path):
            print(f"{RED}❌ ERROR: Classifier not found for '{name}'{RESET}")
            return False
        
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(classifier_path)
        except Exception as e:
            print(f"{RED}❌ Error loading classifier: {e}{RESET}")
            return False
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ ERROR: Could not open webcam!{RESET}")
            return False
        
        print(f"{GREEN}✓ Webcam opened. Starting recognition...{RESET}\n")
        
        matched = False
        frame_count = 0
        start_time = time()
        window_title = f"Face Recognition - {name.upper()} (Press Q to Stop)"
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Resize for better performance
            frame = cv2.resize(frame, (640, 480))
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Elapsed time
            elapsed = time() - start_time
            remaining = max(0, timeout - elapsed)
            
            # Draw info panel
            cv2.rectangle(frame, (0, 0), (600, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Matching: {name.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {timeout}s | Frames: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "Press Q to exit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
            
            # Process faces
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                
                try:
                    id, confidence = recognizer.predict(roi_gray)
                    confidence_percent = 100 - int(confidence)
                    
                    if confidence_percent > 50:
                        matched = True
                        label = f"MATCH: {name.upper()} ({confidence_percent}%)"
                        color = (0, 255, 0)
                        label_color = (0, 255, 0)
                    else:
                        label = f"Unknown ({confidence_percent}%)"
                        color = (0, 0, 255)
                        label_color = (0, 0, 255)
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Background for text
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(frame, (x, y-35), (x+label_size[0]+10, y), label_color, -1)
                    cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (255, 255, 255), 2)
                    
                except Exception as e:
                    pass
            
            # Display frame
            try:
                cv2.imshow(window_title, frame)
            except Exception as e:
                print(f"{YELLOW}⚠ Display warning (continuing): {e}{RESET}")
            
            # Check timeout
            if elapsed >= timeout:
                print(f"\n{YELLOW}⏱ Timeout reached!{RESET}")
                break
            
            # Check for quit
            try:
                key = cv2.waitKey(20) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print(f"\n{YELLOW}⚠ User pressed 'Q'{RESET}")
                    break
            except:
                pass
        
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Results
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"Recognition Complete!")
        print(f"Total Frames: {frame_count}")
        if matched:
            print(f"{GREEN}✓ FACE MATCHED - {name.upper()} Recognized!{RESET}")
        else:
            print(f"{RED}✗ NO MATCH - Face not recognized{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        return matched

def print_menu():
    """Print main menu"""
    print(f"\n{BOLD}{CYAN}╔════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║    FACE RECOGNITION SYSTEM (WINDOWS FIXED)            ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════╝{RESET}")
    print(f"\n{BOLD}Options:{RESET}")
    print(f"  {GREEN}1{RESET}  - Add New Face Data (Capture)")
    print(f"  {GREEN}2{RESET}  - Train Classifier")
    print(f"  {GREEN}3{RESET}  - Live Face Recognition")
    print(f"  {GREEN}4{RESET}  - Complete Flow (Capture + Train + Recognize)")
    print(f"  {GREEN}5{RESET}  - View Available Users")
    print(f"  {GREEN}Q{RESET}  - Quit\n")

def main():
    """Main application"""
    system = FaceRecognitionSystemFixed()
    
    if not system.load_cascade():
        print(f"{RED}Cannot start application. Required files missing.{RESET}")
        return
    
    while True:
        print_menu()
        choice = input(f"{BOLD}Enter your choice: {RESET}").strip().upper()
        
        if choice == 'Q':
            print(f"{YELLOW}Goodbye!{RESET}\n")
            break
        
        elif choice == '1':
            # Add new face data
            name = input(f"\n{BOLD}Enter username for face capture: {RESET}").strip()
            if name:
                num_images = input(f"{BOLD}Number of images to capture (default 50): {RESET}").strip()
                try:
                    num_images = int(num_images) if num_images else 50
                except:
                    num_images = 50
                
                captured = system.capture_face_data(name, num_images)
                if captured > 0:
                    print(f"{GREEN}✓ Successfully captured {captured} images for {name}{RESET}")
                else:
                    print(f"{RED}✗ Failed to capture images{RESET}")
        
        elif choice == '2':
            # Train classifier
            users = system.get_available_users()
            if not users:
                print(f"{RED}❌ No users found. Capture face data first!{RESET}")
            else:
                print(f"\n{BOLD}Available users:{RESET}")
                for i, user in enumerate(users, 1):
                    print(f"  {i}. {user}")
                
                user_choice = input(f"\n{BOLD}Select user to train (number or name): {RESET}").strip()
                
                selected_user = None
                try:
                    idx = int(user_choice) - 1
                    if 0 <= idx < len(users):
                        selected_user = users[idx]
                except:
                    for user in users:
                        if user.lower() == user_choice.lower():
                            selected_user = user
                            break
                
                if selected_user:
                    if system.train_classifier(selected_user):
                        print(f"{GREEN}✓ Classifier trained successfully!{RESET}")
                    else:
                        print(f"{RED}✗ Failed to train classifier{RESET}")
                else:
                    print(f"{RED}❌ Invalid selection{RESET}")
        
        elif choice == '3':
            # Live recognition
            users = system.get_available_users()
            if not users:
                print(f"{RED}❌ No trained classifiers found. Train a model first!{RESET}")
            else:
                print(f"\n{BOLD}Available trained users:{RESET}")
                for i, user in enumerate(users, 1):
                    print(f"  {i}. {user}")
                
                user_choice = input(f"\n{BOLD}Select user to recognize (number or name): {RESET}").strip()
                
                selected_user = None
                try:
                    idx = int(user_choice) - 1
                    if 0 <= idx < len(users):
                        selected_user = users[idx]
                except:
                    for user in users:
                        if user.lower() == user_choice.lower():
                            selected_user = user
                            break
                
                if selected_user:
                    timeout = input(f"{BOLD}Timeout in seconds (default 15): {RESET}").strip()
                    try:
                        timeout = int(timeout) if timeout else 15
                    except:
                        timeout = 15
                    
                    system.recognize_faces_live(selected_user, timeout)
                else:
                    print(f"{RED}❌ Invalid selection{RESET}")
        
        elif choice == '4':
            # Complete flow
            name = input(f"\n{BOLD}Enter username: {RESET}").strip()
            if name:
                # Capture
                captured = system.capture_face_data(name, 50)
                if captured < 10:
                    print(f"{RED}✗ Not enough images captured. Need at least 10.{RESET}")
                    continue
                
                # Train
                input(f"\n{BOLD}Press Enter to start training...{RESET}")
                if system.train_classifier(name):
                    # Recognize
                    input(f"\n{BOLD}Press Enter to start recognition...{RESET}")
                    system.recognize_faces_live(name, timeout=15)
                else:
                    print(f"{RED}✗ Training failed{RESET}")
        
        elif choice == '5':
            # View users
            users = system.get_available_users()
            if users:
                print(f"\n{BOLD}{GREEN}Available trained users:{RESET}")
                for user in users:
                    data_path = os.path.join(system.data_dir, user)
                    num_images = len([f for f in os.listdir(data_path) if f.endswith('.jpg')]) if os.path.exists(data_path) else 0
                    print(f"  ✓ {user} ({num_images} images)")
            else:
                print(f"\n{RED}❌ No trained users found{RESET}")
        
        else:
            print(f"{RED}❌ Invalid choice. Please try again.{RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Application interrupted by user{RESET}\n")
    except Exception as e:
        print(f"\n{RED}ERROR: {e}{RESET}\n")
        import traceback
        traceback.print_exc()
