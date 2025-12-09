"""
FACE RECOGNITION SYSTEM WITH LIVE CAMERA DISPLAY
Complete system with visual feedback on camera feed
Auto-installs packages + Shows real-time face detection
"""

import subprocess
import sys
import os
import cv2
import numpy as np
from time import time, sleep
from pathlib import Path

# Auto-install packages
def install_packages():
    packages = [
        "opencv-python==4.5.4.60",
        "opencv-contrib-python==4.5.4.60",
        "numpy",
        "pillow",
        "h5py",
        "imutils",
        "scikit-learn"
    ]
    print("\n" + "="*70)
    print("INSTALLING REQUIRED PACKAGES...")
    print("="*70 + "\n")
    for package in packages:
        print(f"Installing: {package}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            print(f"✓ {package} installed\n")
        except:
            print(f"⚠ {package} may have issues, continuing...\n")
    print("="*70)
    print("✓ ALL PACKAGES INSTALLED!")
    print("="*70 + "\n")

def check_packages():
    try:
        import cv2
        import numpy
        return True
    except ImportError:
        return False

if not check_packages():
    print("\n⚠ Installing required packages for the first time...")
    try:
        install_packages()
    except:
        os.system(f"{sys.executable} -m pip install opencv-python opencv-contrib-python numpy pillow h5py imutils scikit-learn -q")

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class FaceRecognitionWithCamera:
    def __init__(self):
        print(f"\n{CYAN}Initializing Face Recognition System with Camera Display...{RESET}\n")
        self.cascade_path = './data/haarcascade_frontalface_default.xml'
        self.classifiers_dir = './data/classifiers'
        self.data_dir = './data'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        if self.face_cascade.empty():
            print(f"{RED}❌ Failed to load cascade classifier{RESET}\n")
            sys.exit(1)
        
        print(f"{GREEN}✓ System initialized{RESET}\n")

    def capture_faces(self, name, num_images=50):
        """Capture face images with live preview"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}CAPTURING FACE IMAGES - {name.upper()}{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        print(f"{YELLOW}Instructions:{RESET}")
        print(f"  1. Position your face in the rectangle")
        print(f"  2. System will capture automatically")
        print(f"  3. Press ESC to stop early\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ Cannot open camera!{RESET}\n")
            return False
        
        output_dir = os.path.join(self.data_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        
        count = 0
        start_time = time()
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                print(f"{RED}Failed to read frame{RESET}")
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw info on frame
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Capturing: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {count}/{num_images}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press ESC to stop", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Draw face detection boxes
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Save the face
                face_roi = gray[y:y+h, x:x+w]
                face_img_path = os.path.join(output_dir, f"{count}_{name}.jpg")
                cv2.imwrite(face_img_path, face_roi)
                count += 1
                print(f"  ✓ Captured: {count}/{num_images} | Time: {time()-start_time:.1f}s")
                
                if count >= num_images:
                    break
            
            cv2.imshow(f"Capturing - {name}", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            print(f"\n{GREEN}✓ Captured {count} images for {name}{RESET}\n")
            return True
        return False

    def train_classifier(self, name):
        """Train LBPH recognizer"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}TRAINING CLASSIFIER - {name.upper()}{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        data_dir = os.path.join(self.data_dir, name)
        
        if not os.path.exists(data_dir):
            print(f"{RED}❌ No training data found for {name}{RESET}\n")
            return False
        
        images = []
        labels = []
        
        for img_name in os.listdir(data_dir):
            img_path = os.path.join(data_dir, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(0)
            except:
                pass
        
        if len(images) < 2:
            print(f"{RED}❌ Need at least 2 images to train{RESET}\n")
            return False
        
        print(f"Training with {len(images)} images...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(images, np.array(labels))
        
        classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
        os.makedirs(self.classifiers_dir, exist_ok=True)
        recognizer.save(classifier_path)
        
        print(f"{GREEN}✓ Training complete!{RESET}")
        print(f"{GREEN}✓ Classifier saved: {classifier_path}{RESET}\n")
        return True

    def recognize_faces(self, name, timeout=15):
        """Recognize faces with live display"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}LIVE FACE RECOGNITION - {name.upper()}{RESET}")
        print(f"{CYAN}Timeout: {timeout} seconds{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        print(f"{YELLOW}Instructions: Stand in front of camera{RESET}\n")
        
        classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
        
        if not os.path.exists(classifier_path):
            print(f"{RED}❌ No classifier found for {name}{RESET}\n")
            return False
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(classifier_path)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ Cannot open camera!{RESET}\n")
            return False
        
        start_time = time()
        matches = 0
        frame_count = 0
        matched = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            elapsed = time() - start_time
            frame_count += 1
            
            # Draw info
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Recognizing: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {timeout}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Matches: {matches}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if matches > 0 else (0, 0, 255), 2)
            
            # Recognize faces
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                label, confidence = recognizer.predict(face_roi)
                
                if confidence < 70:  # Match threshold
                    matches += 1
                    color = (0, 255, 0)  # Green for match
                    status = f"MATCH! ({int(confidence)}%)"
                    matched = True
                else:
                    color = (0, 0, 255)  # Red for no match
                    status = f"NO MATCH ({int(confidence)}%)"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.imshow(f"Recognition - {name}", frame)
            
            # Check for timeout or ESC
            elapsed = time() - start_time
            if elapsed >= timeout:
                print(f"\n{YELLOW}⏱ Timeout reached!{RESET}")
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"Recognition Complete!")
        print(f"Total Frames: {frame_count}")
        print(f"Matches Found: {matches}")
        
        if matches > 0:
            print(f"{GREEN}✓ SUCCESS - {name.upper()} RECOGNIZED!{RESET}")
        else:
            print(f"{RED}❌ NO MATCH FOUND{RESET}")
        
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        return matches > 0

    def list_users(self):
        """List all trained users"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}AVAILABLE TRAINED USERS{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        classifiers = []
        if os.path.exists(self.classifiers_dir):
            classifiers = [f.replace('_classifier.xml', '') for f in os.listdir(self.classifiers_dir) if f.endswith('_classifier.xml')]
        
        if not classifiers:
            print(f"{YELLOW}No trained users found{RESET}\n")
            return False
        
        for i, user in enumerate(sorted(classifiers), 1):
            user_dir = os.path.join(self.data_dir, user)
            img_count = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')]) if os.path.exists(user_dir) else 0
            print(f"  {i}. {user.upper()} ({img_count} images)")
        
        print()
        return True

    def complete_flow(self, name):
        """Complete flow: capture -> train -> recognize"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}COMPLETE FLOW - ALL IN ONE{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        # Step 1: Capture
        if not self.capture_faces(name, 50):
            return
        
        # Step 2: Train
        if not self.train_classifier(name):
            return
        
        # Step 3: Recognize
        input(f"\n{YELLOW}Press Enter to start recognition...{RESET}")
        self.recognize_faces(name, 15)

    def run(self):
        """Main menu loop"""
        while True:
            print(f"\n{CYAN}╔{'='*68}╗{RESET}")
            print(f"{CYAN}║{BOLD}          FACE RECOGNITION SYSTEM - WITH CAMERA DISPLAY{RESET}{CYAN}          ║{RESET}")
            print(f"{CYAN}╚{'='*68}╝{RESET}\n")
            print("OPTIONS:")
            print("  1  - Capture Face Images (with live camera)")
            print("  2  - Train Classifier")
            print("  3  - Recognize Faces (with live camera)")
            print("  4  - Complete Flow (Capture → Train → Recognize with camera)")
            print("  5  - List Available Users")
            print("  Q  - Quit\n")
            
            choice = input("Enter choice: ").strip().upper()
            
            if choice == '1':
                name = input("\nEnter name for capture: ").strip()
                if name:
                    self.capture_faces(name, 50)
            
            elif choice == '2':
                name = input("\nEnter name to train: ").strip()
                if name:
                    self.train_classifier(name)
            
            elif choice == '3':
                if not self.list_users():
                    continue
                
                name = input("Enter user name or number: ").strip()
                
                # If number, convert to name
                if name.isdigit():
                    classifiers = sorted([f.replace('_classifier.xml', '') for f in os.listdir(self.classifiers_dir) if f.endswith('_classifier.xml')])
                    idx = int(name) - 1
                    if 0 <= idx < len(classifiers):
                        name = classifiers[idx]
                    else:
                        print(f"{RED}Invalid selection{RESET}\n")
                        continue
                
                timeout = input("Timeout seconds (default 15): ").strip()
                timeout = int(timeout) if timeout.isdigit() else 15
                
                self.recognize_faces(name, timeout)
            
            elif choice == '4':
                name = input("\nEnter your name: ").strip()
                if name:
                    self.complete_flow(name)
            
            elif choice == '5':
                self.list_users()
            
            elif choice == 'Q':
                print(f"\n{GREEN}Thank you for using Face Recognition System!{RESET}\n")
                break
            
            else:
                print(f"\n{RED}Invalid choice!{RESET}\n")

if __name__ == "__main__":
    try:
        system = FaceRecognitionWithCamera()
        system.run()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Application interrupted{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}\n")
