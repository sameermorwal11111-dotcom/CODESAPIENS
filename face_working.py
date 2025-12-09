"""
Face Recognition System - NO GUI DISPLAY ISSUES
Captures images and does recognition WITHOUT cv2.imshow()
Works perfectly on Windows!
"""

import cv2
import os
import numpy as np
from time import time, sleep
import sys

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class FaceRecognitionWorking:
    """Face Recognition - No Display Issues"""
    
    def __init__(self):
        self.cascade_path = './data/haarcascade_frontalface_default.xml'
        self.classifiers_dir = './data/classifiers'
        self.data_dir = './data'
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
    
    def get_available_users(self):
        """Get list of trained users"""
        if not os.path.exists(self.classifiers_dir):
            return []
        users = []
        for file in os.listdir(self.classifiers_dir):
            if file.endswith("_classifier.xml"):
                users.append(file.replace("_classifier.xml", ""))
        return sorted(users)
    
    def capture_face_data(self, name, num_images=50):
        """Capture face images - NO DISPLAY ISSUES"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}{BOLD}CAPTURING {num_images} IMAGES FOR: {name.upper()}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        print(f"{YELLOW}Instructions:{RESET}")
        print("1. Stand in front of camera")
        print("2. Move around - change angles")
        print("3. Vary expressions - smile, neutral")
        print("4. Capture will run for ~30 seconds")
        print("5. Images are being saved automatically")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        path = os.path.join(self.data_dir, name)
        os.makedirs(path, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ ERROR: Webcam not found!{RESET}")
            return 0
        
        print(f"{GREEN}✓ Webcam opened{RESET}")
        print(f"{BLUE}Starting in 3 seconds...{RESET}\n")
        sleep(3)
        
        num_captured = 0
        frame_count = 0
        start_time = time()
        
        while num_captured < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Every 10 frames, show progress
            if frame_count % 10 == 0:
                elapsed = time() - start_time
                print(f"Progress: {num_captured}/{num_images} images | Time: {elapsed:.1f}s | Frames: {frame_count}")
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    img_path = os.path.join(path, f"{num_captured}_{name}.jpg")
                    cv2.imwrite(img_path, roi_gray)
                    num_captured += 1
                    
                    if num_captured % 10 == 0:
                        print(f"{GREEN}✓{RESET} Captured: {num_captured}/{num_images}")
                    
                    if num_captured >= num_images:
                        break
            
            # Press Escape to stop (check every frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print(f"\n{YELLOW}⚠ Stopped by user{RESET}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{GREEN}✓ CAPTURE COMPLETE!{RESET}")
        print(f"  Images captured: {num_captured}")
        print(f"  Total frames: {frame_count}")
        print(f"  Location: {path}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        return num_captured
    
    def train_classifier(self, name):
        """Train face recognizer"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}{BOLD}TRAINING CLASSIFIER FOR: {name.upper()}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        data_path = os.path.join(self.data_dir, name)
        if not os.path.exists(data_path):
            print(f"{RED}❌ No data found for {name}{RESET}")
            return False
        
        images = []
        labels = []
        
        image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        
        if len(image_files) < 10:
            print(f"{RED}❌ Need at least 10 images, found {len(image_files)}{RESET}")
            return False
        
        print(f"{BLUE}Loading {len(image_files)} images...{RESET}\n")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(data_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(0)
                if (i + 1) % 10 == 0:
                    print(f"{GREEN}✓{RESET} Loaded: {i + 1}/{len(image_files)}")
        
        if len(images) < 10:
            print(f"{RED}❌ Could not load enough images{RESET}")
            return False
        
        print(f"\n{BLUE}Training LBPH recognizer...{RESET}\n")
        
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(labels))
            
            os.makedirs(self.classifiers_dir, exist_ok=True)
            classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
            recognizer.write(classifier_path)
            
            print(f"{GREEN}✓ Training complete!{RESET}")
            print(f"  Classifier saved: {classifier_path}")
            print(f"{CYAN}{'='*60}{RESET}\n")
            return True
        except Exception as e:
            print(f"{RED}❌ Training error: {e}{RESET}")
            return False
    
    def recognize_faces(self, name, timeout=15):
        """Live face recognition - NO DISPLAY ERRORS"""
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{CYAN}{BOLD}LIVE FACE RECOGNITION: {name.upper()}{RESET}")
        print(f"{CYAN}Timeout: {timeout} seconds{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
        if not os.path.exists(classifier_path):
            print(f"{RED}❌ Classifier not found for {name}{RESET}")
            return False
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(classifier_path)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ Webcam not found!{RESET}")
            return False
        
        print(f"{GREEN}✓ Webcam opened{RESET}")
        print(f"{BLUE}Starting recognition...{RESET}\n")
        
        matched = False
        frame_count = 0
        match_count = 0
        start_time = time()
        
        print(f"Running for {timeout} seconds...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            elapsed = time() - start_time
            
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                print(f"Time: {elapsed:.1f}s / {timeout}s | Frames: {frame_count} | Status: {'MATCHED' if matched else 'Searching...'}")
            
            # Process detected faces
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                try:
                    id, confidence = recognizer.predict(roi_gray)
                    confidence_percent = 100 - int(confidence)
                    
                    if confidence_percent > 50:
                        matched = True
                        match_count += 1
                        if match_count % 10 == 0:
                            print(f"{GREEN}✓ MATCH DETECTED!{RESET} Confidence: {confidence_percent}%")
                except:
                    pass
            
            # Check timeout
            if elapsed >= timeout:
                print(f"\n{YELLOW}⏱ Timeout reached!{RESET}")
                break
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                print(f"\n{YELLOW}⚠ Stopped by user{RESET}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Results
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"Recognition Complete!")
        print(f"Total Frames: {frame_count}")
        print(f"Matches Found: {match_count}")
        if matched:
            print(f"{GREEN}✓ SUCCESS - {name.upper()} RECOGNIZED!{RESET}")
        else:
            print(f"{RED}✗ NO MATCH - Face not recognized{RESET}")
        print(f"{CYAN}{'='*60}{RESET}\n")
        
        return matched

def print_menu():
    """Print menu"""
    print(f"\n{BOLD}{CYAN}╔════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║     FACE RECOGNITION SYSTEM - WORKING VERSION          ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════╝{RESET}")
    print(f"\n{BOLD}Options:{RESET}")
    print(f"  {GREEN}1{RESET}  - Capture Face Images")
    print(f"  {GREEN}2{RESET}  - Train Classifier")
    print(f"  {GREEN}3{RESET}  - Recognize Faces")
    print(f"  {GREEN}4{RESET}  - Complete Flow (Capture → Train → Recognize)")
    print(f"  {GREEN}5{RESET}  - List Available Users")
    print(f"  {GREEN}Q{RESET}  - Quit\n")

def main():
    """Main"""
    system = FaceRecognitionWorking()
    
    while True:
        print_menu()
        choice = input(f"{BOLD}Enter choice: {RESET}").strip().upper()
        
        if choice == 'Q':
            print(f"{YELLOW}Goodbye!{RESET}\n")
            break
        
        elif choice == '1':
            name = input(f"\n{BOLD}Enter username: {RESET}").strip()
            if name:
                num = input(f"{BOLD}Number of images (default 50): {RESET}").strip()
                try:
                    num = int(num) if num else 50
                except:
                    num = 50
                
                captured = system.capture_face_data(name, num)
                if captured > 0:
                    print(f"{GREEN}✓ Captured {captured} images{RESET}")
        
        elif choice == '2':
            users = system.get_available_users()
            if not users:
                print(f"{RED}❌ No users found{RESET}")
            else:
                print(f"\n{BOLD}Available users:{RESET}")
                for i, user in enumerate(users, 1):
                    print(f"  {i}. {user}")
                
                sel = input(f"\n{BOLD}Select user (number or name): {RESET}").strip()
                
                user = None
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(users):
                        user = users[idx]
                except:
                    for u in users:
                        if u.lower() == sel.lower():
                            user = u
                            break
                
                if user:
                    if system.train_classifier(user):
                        print(f"{GREEN}✓ Training successful!{RESET}")
        
        elif choice == '3':
            users = system.get_available_users()
            if not users:
                print(f"{RED}❌ No trained users{RESET}")
            else:
                print(f"\n{BOLD}Available users:{RESET}")
                for i, user in enumerate(users, 1):
                    print(f"  {i}. {user}")
                
                sel = input(f"\n{BOLD}Select user (number or name): {RESET}").strip()
                
                user = None
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(users):
                        user = users[idx]
                except:
                    for u in users:
                        if u.lower() == sel.lower():
                            user = u
                            break
                
                if user:
                    tout = input(f"{BOLD}Timeout seconds (default 15): {RESET}").strip()
                    try:
                        tout = int(tout) if tout else 15
                    except:
                        tout = 15
                    
                    system.recognize_faces(user, tout)
        
        elif choice == '4':
            name = input(f"\n{BOLD}Enter username: {RESET}").strip()
            if name:
                # Capture
                captured = system.capture_face_data(name, 50)
                if captured < 10:
                    print(f"{RED}✗ Need at least 10 images{RESET}")
                    continue
                
                # Train
                input(f"\n{BOLD}Press Enter to train...{RESET}")
                if system.train_classifier(name):
                    # Recognize
                    input(f"\n{BOLD}Press Enter to recognize...{RESET}")
                    system.recognize_faces(name, 15)
        
        elif choice == '5':
            users = system.get_available_users()
            if users:
                print(f"\n{BOLD}{GREEN}Trained users:{RESET}")
                for user in users:
                    data_path = os.path.join(system.data_dir, user)
                    num = len([f for f in os.listdir(data_path) if f.endswith('.jpg')]) if os.path.exists(data_path) else 0
                    print(f"  ✓ {user} ({num} images)")
            else:
                print(f"\n{RED}No trained users{RESET}")
        
        else:
            print(f"{RED}Invalid choice{RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted{RESET}\n")
    except Exception as e:
        print(f"\n{RED}ERROR: {e}{RESET}\n")
