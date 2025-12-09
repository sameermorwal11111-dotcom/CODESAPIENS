"""
FACE RECOGNITION SYSTEM WITH LIVE CAMERA DISPLAY
Complete system with visual feedback + Auto-detect all users
Auto-installs packages + Shows real-time face detection + Names on faces
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

class FaceRecognitionWithAutoDetect:
    def __init__(self):
        print(f"\n{CYAN}Initializing Face Recognition System with Auto-Detection...{RESET}\n")
        self.cascade_path = './data/haarcascade_frontalface_default.xml'
        self.classifiers_dir = './data/classifiers'
        self.data_dir = './data'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        
        if self.face_cascade.empty():
            print(f"{RED}❌ Failed to load cascade classifier{RESET}\n")
            sys.exit(1)
        
        # Load all classifiers for auto-detection
        self.recognizers = {}
        self.load_all_recognizers()
        
        print(f"{GREEN}✓ System initialized with auto-detection{RESET}\n")

    def load_all_recognizers(self):
        """Load all trained recognizers"""
        if not os.path.exists(self.classifiers_dir):
            print(f"{YELLOW}⚠ No classifiers found yet{RESET}\n")
            return
        
        for filename in os.listdir(self.classifiers_dir):
            if filename.endswith('_classifier.xml'):
                user_name = filename.replace('_classifier.xml', '')
                classifier_path = os.path.join(self.classifiers_dir, filename)
                try:
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(classifier_path)
                    self.recognizers[user_name] = recognizer
                    print(f"✓ Loaded: {user_name}")
                except:
                    pass

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
            for (x, y, w_face, h_face) in faces:
                cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), (0, 255, 0), 2)
                # Save the face
                face_roi = gray[y:y+h_face, x:x+w_face]
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
        
        # Reload recognizers
        self.load_all_recognizers()
        
        print(f"{GREEN}✓ Training complete!{RESET}")
        print(f"{GREEN}✓ Classifier saved: {classifier_path}{RESET}\n")
        return True

    def recognize_faces_auto(self, timeout=15):
        """Recognize ALL faces with live display and show names"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}LIVE FACE RECOGNITION - AUTO-DETECT ALL USERS{RESET}")
        print(f"{CYAN}Timeout: {timeout} seconds{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        print(f"{YELLOW}Instructions: Stand in front of camera{RESET}\n")
        
        if not self.recognizers:
            print(f"{RED}❌ No trained recognizers found!{RESET}")
            print(f"{YELLOW}Please train at least one user first.{RESET}\n")
            return False
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ Cannot open camera!{RESET}\n")
            return False
        
        start_time = time()
        matches = {}  # {person_name: match_count}
        frame_count = 0
        
        # Initialize match counters
        for user_name in self.recognizers.keys():
            matches[user_name] = 0
        
        print(f"{GREEN}Starting recognition...{RESET}\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            elapsed = time() - start_time
            frame_count += 1
            
            # Draw info on frame
            h, w = frame.shape[:2]
            cv2.putText(frame, "AUTO-DETECT ALL USERS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {timeout}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {frame_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Recognize faces
            recognized_any = False
            for (x, y, w_face, h_face) in faces:
                face_roi = gray[y:y+h_face, x:x+w_face]
                
                best_match = None
                best_confidence = 100
                
                # Try to match against all recognizers
                for user_name, recognizer in self.recognizers.items():
                    label, confidence = recognizer.predict(face_roi)
                    
                    # Lower confidence = better match
                    if confidence < best_confidence:
                        best_confidence = confidence
                        best_match = user_name
                
                # If we found a match
                if best_match and best_confidence < 70:
                    color = (0, 255, 0)  # Green for match
                    status = f"{best_match.upper()} ({int(best_confidence)}%)"
                    matches[best_match] += 1
                    recognized_any = True
                else:
                    color = (0, 0, 255)  # Red for no clear match
                    status = f"UNKNOWN ({int(best_confidence)}%)"
                
                cv2.rectangle(frame, (x, y), (x+w_face, y+h_face), color, 2)
                cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw match summary at bottom
            match_text = "Matches: " + " | ".join([f"{name}:{count}" for name, count in matches.items() if count > 0])
            if match_text != "Matches: ":
                cv2.putText(frame, match_text, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            cv2.imshow("Recognition - Auto-Detect All Users", frame)
            
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
        
        # Show results
        print(f"\nResults:")
        total_matches = sum(matches.values())
        for user_name, count in matches.items():
            if count > 0:
                print(f"  {user_name.upper()}: {count} matches ✓")
        
        if total_matches > 0:
            print(f"\n{GREEN}✓ RECOGNIZED {total_matches} FACE(S)!{RESET}")
        else:
            print(f"\n{RED}❌ NO MATCHES FOUND{RESET}")
        
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        return total_matches > 0

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
        input(f"\n{YELLOW}Press Enter to start auto-detection recognition...{RESET}")
        self.recognize_faces_auto(15)

    def run(self):
        """Main menu loop"""
        while True:
            print(f"\n{CYAN}╔{'='*68}╗{RESET}")
            print(f"{CYAN}║{BOLD}    FACE RECOGNITION - WITH AUTO-DETECT & LIVE CAMERA{RESET}{CYAN}    ║{RESET}")
            print(f"{CYAN}╚{'='*68}╝{RESET}\n")
            print("OPTIONS:")
            print("  1  - Capture Face Images (with live camera)")
            print("  2  - Train Classifier")
            print("  3  - Recognize ALL Faces (auto-detect, no asking)")
            print("  4  - Complete Flow (Capture → Train → Auto-Recognize)")
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
                # Auto-detect mode - no asking for name
                timeout = input("Timeout seconds (default 15): ").strip()
                timeout = int(timeout) if timeout.isdigit() else 15
                self.recognize_faces_auto(timeout)
            
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
        system = FaceRecognitionWithAutoDetect()
        system.run()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Application interrupted{RESET}\n")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}\n")
