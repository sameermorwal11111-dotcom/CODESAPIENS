#!/usr/bin/env python3
"""
ADVANCED FACE RECOGNITION WITH LIVE WEBCAM - ENHANCED VERSION
More accurate matching + Larger display + 100 default captures
+ User-selectable recognition timeout
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
B = '\033[1m'    # Bold

class AdvancedFaceRecognition:
    def __init__(self):
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}ADVANCED FACE RECOGNITION - ENHANCED ACCURACY{Z}")
        print(f"{C}{'='*70}{Z}\n")
        
        self.cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        if self.cascade.empty():
            print(f"{R}ERROR: Cascade file not found!{Z}\n")
            sys.exit(1)
        
        self.recognizers = {}
        self.confidence_data = {}  # Store confidence scores for better accuracy
        self.load_recognizers()
        print(f"{G}✓ System ready!{Z}\n")
    
    def load_recognizers(self):
        """Load all trained classifiers with enhanced accuracy"""
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
                    self.confidence_data[name] = {'threshold': 70, 'matches': 0}
                    print(f"  ✓ Loaded: {name}")
                except:
                    pass
        
        if self.recognizers:
            print(f"\n{G}✓ {len(self.recognizers)} trained users loaded{Z}\n")
    
    def capture_faces(self, name, num=100):
        """Capture faces with LARGE LIVE WEBCAM DISPLAY - Enhanced"""
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}CAPTURING FACE IMAGES - {name.upper()}{Z}")
        print(f"{C}{'='*70}{Z}\n")
        
        print(f"{Y}Target: {num} images for better accuracy{Z}")
        print(f"{Y}Close distance recommended for better training{Z}\n")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{R}ERROR: Cannot open camera!{Z}")
            print(f"{Y}Solutions:{Z}")
            print(f"  1. Close other apps using camera (Zoom, Skype, etc)")
            print(f"  2. Restart your computer")
            print(f"  3. Check Windows camera permissions\n")
            return False
        
        # Set LARGE window resolution for better display
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Create output directory
        out_dir = f'./data/{name}'
        os.makedirs(out_dir, exist_ok=True)
        
        count = 0
        start = time()
        window = f"CAPTURE - {name.upper()} [PRESS ESC TO STOP]"
        
        print(f"{G}✓ Webcam OPENED - LARGE WINDOW DISPLAYED{Z}")
        print(f"{Y}Position your face in center - closer is better{Z}")
        print(f"{Y}System will auto-capture as it detects faces{Z}\n")
        
        face_size_threshold = 80  # Minimum face size for quality
        
        while count < num:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced face detection with size filtering
            faces = self.cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,      # More accurate
                minNeighbors=6,       # Stricter (more accurate)
                minSize=(face_size_threshold, face_size_threshold),
                maxSize=(400, 400)
            )
            
            h, w = frame.shape[:2]
            
            # Dark background for better contrast
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 40), -1)
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            
            # LARGE TITLE
            cv2.putText(frame, f"CAPTURING - {name.upper()}", (40, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            
            # Progress bar
            progress_percent = (count / num) * 100
            bar_width = int((w - 80) * progress_percent / 100)
            cv2.rectangle(frame, (40, 140), (w-40, 180), (100, 100, 100), 2)
            cv2.rectangle(frame, (40, 140), (40+bar_width, 180), (0, 255, 0), -1)
            cv2.putText(frame, f"Progress: {count}/{num} ({progress_percent:.1f}%)", (50, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # Time and FPS
            elapsed = time() - start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"Time: {elapsed:.1f}s | FPS: {fps:.1f}", (50, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            
            # Face detection status
            if len(faces) > 0:
                status_color = (0, 255, 0)
                status_text = f"✓ {len(faces)} face(s) detected - CAPTURING"
            else:
                status_color = (0, 0, 255)
                status_text = "⚠ Position face in center"
            
            cv2.putText(frame, status_text, (50, h-80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
            
            cv2.putText(frame, "Press ESC to stop | Stand closer for better quality", (50, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 255), 2)
            
            # Detect and save faces
            for (x, y, ww, hh) in faces:
                # THICK GREEN RECTANGLE
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 5)
                
                # Draw circle in center of face
                center_x, center_y = x + ww//2, y + hh//2
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Extract and save face with better quality
                face_roi = gray[y:y+hh, x:x+ww]
                face_img_path = f'{out_dir}/{count}_{name}.jpg'
                
                # Save with high quality
                cv2.imwrite(face_img_path, face_roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                count += 1
                print(f"  ✓ Captured: {count}/{num} ({(count/num)*100:.1f}%) - Quality: HIGH")
                
                if count >= num:
                    break
            
            frame_count += 1
            
            # SHOW LARGE WINDOW
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print(f"\n{Y}⏹ Stopped by user - {count} images captured{Z}\n")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        
        if count > 0:
            print(f"{G}{'='*70}{Z}")
            print(f"{G}✓ CAPTURE COMPLETE!{Z}")
            print(f"{G}  Total Images: {count}")
            print(f"  Directory: ./data/{name}/")
            print(f"  Quality: HIGH (95% JPEG)")
            print(f"{G}{'='*70}{Z}\n")
            return True
        return False
    
    def train_classifier(self, name):
        """Train classifier with enhanced accuracy"""
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}TRAINING CLASSIFIER - {name.upper()}{Z}")
        print(f"{C}{'='*70}{Z}\n")
        
        data_dir = f'./data/{name}'
        if not os.path.exists(data_dir):
            print(f"{R}No images found for {name}{Z}\n")
            return False
        
        images = []
        labels = []
        
        print(f"{Y}Loading images...{Z}")
        for img_name in os.listdir(data_dir):
            if img_name.endswith('.jpg'):
                img = cv2.imread(os.path.join(data_dir, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Enhance image quality for better training
                    img = cv2.equalizeHist(img)  # Histogram equalization
                    images.append(img)
                    labels.append(0)
        
        if len(images) < 2:
            print(f"{R}Need at least 2 images{Z}\n")
            return False
        
        print(f"{Y}Training with {len(images)} images...{Z}")
        
        # Enhanced LBPH recognizer with better parameters
        rec = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8
        )
        rec.train(images, np.array(labels))
        
        os.makedirs('./data/classifiers', exist_ok=True)
        rec.save(f'./data/classifiers/{name}_classifier.xml')
        
        self.load_recognizers()
        
        print(f"{G}{'='*70}{Z}")
        print(f"{G}✓ TRAINING COMPLETE!{Z}")
        print(f"{G}  Model: ./data/classifiers/{name}_classifier.xml")
        print(f"  Images Used: {len(images)}")
        print(f"  Algorithm: LBPH (Enhanced)")
        print(f"  Quality: IMPROVED (Histogram Equalization)")
        print(f"{G}{'='*70}{Z}\n")
        return True
    
    def recognize_all(self, timeout=15):
        """Recognize ALL faces with LARGE DISPLAY and PRECISE accuracy"""
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}LIVE FACE RECOGNITION - AUTO-DETECT ALL USERS{Z}")
        print(f"{C}{'='*70}{Z}\n")
        
        if not self.recognizers:
            print(f"{R}No trained users found!{Z}")
            print(f"{Y}Please train at least one user first (Option 2).{Z}\n")
            return
        
        print(f"{Y}Trained users: {', '.join(self.recognizers.keys())}{Z}\n")
        print(f"{G}Opening webcam... (LARGE WINDOW){Z}\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{R}ERROR: Cannot open camera!{Z}\n")
            return
        
        # LARGE RESOLUTION
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        start = time()
        matches = {name: 0 for name in self.recognizers.keys()}
        frame_count = 0
        window = f"RECOGNITION - ALL USERS [PRESS ESC TO STOP]"
        
        print(f"{G}✓ Webcam OPENED - LARGE DISPLAY{Z}")
        print(f"{Y}Stand in front of camera{Z}")
        print(f"{Y}GREEN box = RECOGNIZED | RED box = UNKNOWN{Z}\n")
        
        confidence_history = {name: [] for name in self.recognizers.keys()}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced detection
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(80, 80),
                maxSize=(400, 400)
            )
            
            elapsed = time() - start
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 40), -1)
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            
            # LARGE TITLE
            cv2.putText(frame, "AUTO-DETECT ALL USERS", (40, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            
            # Progress/Time
            cv2.putText(frame, f"Time: {elapsed:.1f}s / {timeout}s", (40, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Frames: {frame_count}", (40, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 2)
            
            # Process faces with enhanced accuracy
            for (x, y, ww, hh) in faces:
                face_roi = gray[y:y+hh, x:x+ww]
                face_roi = cv2.equalizeHist(face_roi)  # Enhance for better matching
                
                best_name = None
                best_conf = 100
                
                # Compare with all recognizers
                for name, rec in self.recognizers.items():
                    label, conf = rec.predict(face_roi)
                    confidence_history[name].append(conf)
                    
                    # Keep only recent history (last 30 frames)
                    if len(confidence_history[name]) > 30:
                        confidence_history[name].pop(0)
                    
                    if conf < best_conf:
                        best_conf = conf
                        best_name = name
                
                # IMPROVED ACCURACY: Use average confidence + threshold
                if best_name:
                    avg_conf = np.mean(confidence_history[best_name])
                    
                    if avg_conf < 70:
                        # MATCH FOUND
                        color = (0, 255, 0)  # GREEN
                        text = f"{best_name.upper()} ({int(avg_conf)}%)"
                        matches[best_name] += 1
                        border_thickness = 5
                    else:
                        # NO MATCH - RED BOX
                        color = (0, 0, 255)  # RED
                        text = f"UNKNOWN ({int(avg_conf)}%)"
                        border_thickness = 3
                else:
                    color = (0, 0, 255)  # RED
                    text = f"UNKNOWN"
                    border_thickness = 3
                
                # THICK RECTANGLE
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), color, border_thickness)
                
                # Draw larger face label with background
                label_bg_width = len(text) * 12
                cv2.rectangle(frame, (x-2, y-50), (x+label_bg_width+5, y-10), color, -1)
                cv2.putText(frame, text, (x, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
                
                # Center circle
                center_x, center_y = x + ww//2, y + hh//2
                cv2.circle(frame, (center_x, center_y), 5, color, -1)
            
            # Bottom match summary - LARGE
            match_summary = []
            for name, count in matches.items():
                if count > 0:
                    match_summary.append(f"{name}:{count}")
            
            if match_summary:
                summary_text = "MATCHES: " + " | ".join(match_summary)
                cv2.putText(frame, summary_text, (40, h-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            
            # Instructions
            cv2.putText(frame, "Press ESC to stop recognition", (40, h-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 255), 2)
            
            # SHOW LARGE WINDOW
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or elapsed >= timeout:  # ESC or timeout
                if key == 27:
                    print(f"\n{Y}⏹ Stopped by user{Z}")
                else:
                    print(f"\n{Y}⏱ Timeout reached{Z}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}RECOGNITION RESULTS{Z}")
        print(f"{C}{'='*70}{Z}\n")
        print(f"Total Frames Analyzed: {frame_count}")
        print(f"Duration: {elapsed:.1f} seconds\n")
        
        total_matches = sum(matches.values())
        
        print(f"Results:")
        for name, count in matches.items():
            if count > 0:
                print(f"  {B}{name.upper()}{Z}: {count} matches ✓")
            else:
                print(f"  {Y}{name.upper()}{Z}: 0 matches")
        
        if total_matches > 0:
            print(f"\n{G}{B}✓ SUCCESSFULLY RECOGNIZED {total_matches} FACE(S)!{Z}\n")
        else:
            print(f"\n{R}❌ NO FACES MATCHED - ALL MARKED AS UNKNOWN{Z}\n")
        
        print(f"{C}{'='*70}{Z}\n")
    
    def list_users(self):
        """List all trained users with statistics"""
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}TRAINED USERS - STATISTICS{Z}")
        print(f"{C}{'='*70}{Z}\n")
        
        classifiers = []
        if os.path.exists('./data/classifiers'):
            classifiers = [f.replace('_classifier.xml', '') for f in os.listdir('./data/classifiers') if f.endswith('_classifier.xml')]
        
        if not classifiers:
            print(f"{Y}No trained users found{Z}\n")
            return False
        
        print(f"Total Users: {len(classifiers)}\n")
        
        for i, user in enumerate(sorted(classifiers), 1):
            user_dir = f'./data/{user}'
            img_count = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')]) if os.path.exists(user_dir) else 0
            
            # File size
            classifier_file = f'./data/classifiers/{user}_classifier.xml'
            size_kb = os.path.getsize(classifier_file) / 1024 if os.path.exists(classifier_file) else 0
            
            print(f"  {i}. {B}{user.upper()}{Z}")
            print(f"     Images: {img_count}")
            print(f"     Model Size: {size_kb:.1f} KB")
            print(f"     Status: ✓ Ready\n")
        
        return True
    
    def complete_flow(self, name):
        """Complete flow: capture -> train -> recognize"""
        print(f"\n{C}{'='*70}{Z}")
        print(f"{C}{B}COMPLETE FLOW - CAPTURE + TRAIN + RECOGNIZE{Z}")
        print(f"{C}{'='*70}{Z}\n")
        
        # Step 1: Capture
        print(f"{B}STEP 1: CAPTURING IMAGES{Z}\n")
        if not self.capture_faces(name, 100):
            return
        
        # Step 2: Train
        print(f"{B}STEP 2: TRAINING CLASSIFIER{Z}\n")
        if not self.train_classifier(name):
            return
        
        # Step 3: Recognize
        print(f"{B}STEP 3: RECOGNIZE (AUTO-DETECT){Z}\n")
        input(f"{Y}Press Enter to start recognition...{Z}")
        self.recognize_all(15)
    
    def menu(self):
        """Main menu with enhanced options"""
        while True:
            print(f"\n{C}{'='*70}{Z}")
            print(f"{C}{B}FACE RECOGNITION SYSTEM - ADVANCED VERSION{Z}")
            print(f"{C}{'='*70}{Z}\n")
            print(f"{Y}Features: Enhanced Accuracy | Large Display | 100 Captures | Custom Timeout{Z}\n")
            print("MENU:")
            print("  1 - Capture Face Images (100 images, LARGE WEBCAM)")
            print("  2 - Train Classifier (Enhanced Accuracy)")
            print("  3 - Recognize All Faces (Custom Timeout, LARGE DISPLAY)")
            print("  4 - Complete Flow (Capture → Train → Recognize)")
            print("  5 - List Trained Users & Statistics")
            print("  Q - Quit\n")
            
            choice = input("Enter choice: ").strip().upper()
            
            if choice == '1':
                name = input("\nEnter name for capture: ").strip()
                if name:
                    self.capture_faces(name, 100)
                else:
                    print(f"{R}Invalid name{Z}\n")
            
            elif choice == '2':
                name = input("\nEnter name to train: ").strip()
                if name:
                    self.train_classifier(name)
                else:
                    print(f"{R}Invalid name{Z}\n")
            
            elif choice == '3':
                print(f"\n{Y}Select recognition timeout:{Z}")
                print("  1 - 10 seconds (Quick)")
                print("  2 - 15 seconds (Normal) - DEFAULT")
                print("  3 - 30 seconds (Extended)")
                print("  4 - 60 seconds (Long)")
                print("  5 - Custom")
                
                t_choice = input("\nEnter choice: ").strip()
                
                if t_choice == '1':
                    timeout = 10
                elif t_choice == '2':
                    timeout = 15
                elif t_choice == '3':
                    timeout = 30
                elif t_choice == '4':
                    timeout = 60
                elif t_choice == '5':
                    try:
                        timeout = int(input("Enter timeout in seconds: ").strip())
                        if timeout < 5:
                            timeout = 5
                            print(f"{Y}Minimum 5 seconds - set to 5{Z}")
                    except:
                        timeout = 15
                        print(f"{Y}Invalid input - using default 15s{Z}")
                else:
                    timeout = 15
                    print(f"{Y}Invalid choice - using default 15s{Z}")
                
                print(f"\n{G}Starting recognition with {timeout}s timeout...{Z}\n")
                self.recognize_all(timeout)
            
            elif choice == '4':
                name = input("\nEnter your name: ").strip()
                if name:
                    self.complete_flow(name)
                else:
                    print(f"{R}Invalid name{Z}\n")
            
            elif choice == '5':
                self.list_users()
            
            elif choice == 'Q':
                print(f"\n{G}{B}Thank you for using Advanced Face Recognition!{Z}\n")
                break
            
            else:
                print(f"\n{R}Invalid choice! Please try again.{Z}\n")

if __name__ == "__main__":
    try:
        app = AdvancedFaceRecognition()
        app.menu()
    except KeyboardInterrupt:
        print(f"\n\n{Y}Application interrupted by user{Z}\n")
    except Exception as e:
        print(f"\n{R}Error: {e}{Z}\n")
        import traceback
        traceback.print_exc()
