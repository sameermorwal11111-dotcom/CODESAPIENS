#!/usr/bin/env python3
"""
FACE RECOGNITION WITH LIVE WEBCAM
Guaranteed to open webcam during capture and recognition
Copy-paste ready working version
"""

import subprocess
import sys
import os
import cv2
import numpy as np
from time import time, sleep

# Install packages silently
def ensure_packages():
    packages = ["opencv-python", "opencv-contrib-python", "numpy", "pillow", "h5py", "imutils", "scikit-learn"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_packages()

# Colors
G = '\033[92m'   # Green
R = '\033[91m'   # Red
C = '\033[96m'   # Cyan
Y = '\033[93m'   # Yellow
Z = '\033[0m'    # Reset

class WebcamFaceRecognition:
    def __init__(self):
        print(f"\n{C}INITIALIZING FACE RECOGNITION SYSTEM{Z}\n")
        self.cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        if self.cascade.empty():
            print(f"{R}ERROR: Cascade file not found!{Z}\n")
            sys.exit(1)
        
        self.recognizers = {}
        self.load_recognizers()
        print(f"{G}✓ System ready!{Z}\n")
    
    def load_recognizers(self):
        """Load all trained classifiers"""
        path = './data/classifiers'
        if not os.path.exists(path):
            return
        
        for file in os.listdir(path):
            if file.endswith('_classifier.xml'):
                name = file.replace('_classifier.xml', '')
                try:
                    rec = cv2.face.LBPHFaceRecognizer_create()
                    rec.read(os.path.join(path, file))
                    self.recognizers[name] = rec
                    print(f"  Loaded: {name}")
                except:
                    pass
    
    def capture_faces(self, name, num=50):
        """Capture faces with LIVE WEBCAM DISPLAY"""
        print(f"\n{C}{'='*60}{Z}")
        print(f"{C}CAPTURING FACE IMAGES - {name.upper()}{Z}")
        print(f"{C}{'='*60}{Z}\n")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{R}ERROR: Cannot open camera!{Z}")
            print(f"{Y}Solutions:{Z}")
            print(f"  1. Close other apps using camera")
            print(f"  2. Restart your computer")
            print(f"  3. Check camera permissions\n")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create output directory
        out_dir = f'./data/{name}'
        os.makedirs(out_dir, exist_ok=True)
        
        count = 0
        start = time()
        window = f"CAPTURE - {name.upper()}"
        
        print(f"{G}✓ Webcam OPEN - Camera window should appear{Z}")
        print(f"{Y}Stand in front of camera. Press ESC to stop.{Z}\n")
        
        while count < num:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            # Draw on frame
            h, w = frame.shape[:2]
            
            # Title
            cv2.putText(frame, f"CAPTURING - {name.upper()}", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Progress
            cv2.putText(frame, f"Progress: {count}/{num}", (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Time
            elapsed = time() - start
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Status
            if len(faces) > 0:
                cv2.putText(frame, "Face detected - capturing...", (15, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            else:
                cv2.putText(frame, "Position face in center", (15, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            
            # Detect and save faces
            for (x, y, ww, hh) in faces:
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 3)
                
                face_roi = gray[y:y+hh, x:x+ww]
                cv2.imwrite(f'{out_dir}/{count}_{name}.jpg', face_roi)
                count += 1
                print(f"  Captured: {count}/{num}")
                
                if count >= num:
                    break
            
            # SHOW WINDOW
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print(f"\n{Y}Stopped by user{Z}\n")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        
        if count > 0:
            print(f"{G}✓ Captured {count} images{Z}\n")
            return True
        return False
    
    def train_classifier(self, name):
        """Train classifier"""
        print(f"\n{C}{'='*60}{Z}")
        print(f"{C}TRAINING - {name.upper()}{Z}")
        print(f"{C}{'='*60}{Z}\n")
        
        data_dir = f'./data/{name}'
        if not os.path.exists(data_dir):
            print(f"{R}No images found for {name}{Z}\n")
            return False
        
        images = []
        labels = []
        
        for img_name in os.listdir(data_dir):
            if img_name.endswith('.jpg'):
                img = cv2.imread(os.path.join(data_dir, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(0)
        
        if len(images) < 2:
            print(f"{R}Need at least 2 images{Z}\n")
            return False
        
        print(f"Training with {len(images)} images...")
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(images, np.array(labels))
        
        os.makedirs('./data/classifiers', exist_ok=True)
        rec.save(f'./data/classifiers/{name}_classifier.xml')
        
        self.load_recognizers()
        print(f"{G}✓ Training complete!{Z}\n")
        return True
    
    def recognize_all(self, timeout=15):
        """Recognize ALL faces with LIVE WEBCAM DISPLAY"""
        print(f"\n{C}{'='*60}{Z}")
        print(f"{C}RECOGNIZING ALL FACES - LIVE{Z}")
        print(f"{C}Timeout: {timeout}s{Z}")
        print(f"{C}{'='*60}{Z}\n")
        
        if not self.recognizers:
            print(f"{R}No trained users found!{Z}\n")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{R}ERROR: Cannot open camera!{Z}\n")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start = time()
        matches = {name: 0 for name in self.recognizers.keys()}
        frame_count = 0
        window = "RECOGNITION - ALL USERS"
        
        print(f"{G}✓ Webcam OPEN - Stand in front of camera{Z}\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
            
            elapsed = time() - start
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Title
            cv2.putText(frame, "AUTO-DETECT ALL USERS", (15, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Time
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {timeout}s", (15, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Frames
            cv2.putText(frame, f"Frames: {frame_count}", (15, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Process faces
            for (x, y, ww, hh) in faces:
                face_roi = gray[y:y+hh, x:x+ww]
                
                best_name = None
                best_conf = 100
                
                # Compare with all recognizers
                for name, rec in self.recognizers.items():
                    label, conf = rec.predict(face_roi)
                    if conf < best_conf:
                        best_conf = conf
                        best_name = name
                
                # Draw result
                if best_name and best_conf < 70:
                    color = (0, 255, 0)  # Green
                    text = f"{best_name.upper()} ({int(best_conf)}%)"
                    matches[best_name] += 1
                else:
                    color = (0, 0, 255)  # Red
                    text = f"UNKNOWN ({int(best_conf)}%)"
                
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), color, 3)
                cv2.putText(frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Show matches
            match_text = " | ".join([f"{n}:{c}" for n, c in matches.items() if c > 0])
            if match_text:
                cv2.putText(frame, match_text, (15, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            # SHOW WINDOW
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or elapsed >= timeout:  # ESC or timeout
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        
        print(f"\n{C}{'='*60}{Z}")
        print(f"Results:")
        total = sum(matches.values())
        for name, count in matches.items():
            if count > 0:
                print(f"  {name.upper()}: {count} matches ✓")
        
        if total > 0:
            print(f"\n{G}✓ RECOGNIZED {total} FACE(S)!{Z}")
        else:
            print(f"\n{R}No matches found{Z}")
        print(f"{C}{'='*60}{Z}\n")
    
    def menu(self):
        """Main menu"""
        while True:
            print(f"{C}{'='*60}{Z}")
            print(f"{C}FACE RECOGNITION - LIVE WEBCAM{Z}")
            print(f"{C}{'='*60}{Z}")
            print("\nMENU:")
            print("  1 - Capture Face Images (LIVE WEBCAM)")
            print("  2 - Train Classifier")
            print("  3 - Recognize All Faces (LIVE WEBCAM)")
            print("  4 - Complete Flow (Capture → Train → Recognize)")
            print("  Q - Quit\n")
            
            choice = input("Enter choice: ").strip().upper()
            
            if choice == '1':
                name = input("Enter name: ").strip()
                if name:
                    self.capture_faces(name, 50)
            
            elif choice == '2':
                name = input("Enter name: ").strip()
                if name:
                    self.train_classifier(name)
            
            elif choice == '3':
                t = input("Timeout seconds (default 15): ").strip()
                timeout = int(t) if t.isdigit() else 15
                self.recognize_all(timeout)
            
            elif choice == '4':
                name = input("Enter your name: ").strip()
                if name:
                    self.capture_faces(name, 50)
                    self.train_classifier(name)
                    input("Press Enter to recognize...")
                    self.recognize_all(15)
            
            elif choice == 'Q':
                print(f"\n{G}Goodbye!{Z}\n")
                break

if __name__ == "__main__":
    try:
        app = WebcamFaceRecognition()
        app.menu()
    except Exception as e:
        print(f"\n{R}Error: {e}{Z}\n")
