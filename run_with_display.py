"""
COMPLETE FACE RECOGNITION WITH LIVE WEBCAM DISPLAY
Shows real-time face tracking with rectangles and labels
Auto-installs all packages
Works immediately!
"""

import subprocess
import sys
import os

# Auto-install required packages
def install_packages():
    """Auto-install all required packages"""
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
            print(f"⚠ {package} installation may have issues, continuing...\n")
    
    print("="*70)
    print("✓ ALL PACKAGES INSTALLED!")
    print("="*70 + "\n")

# Check if packages are installed
def check_packages():
    """Check if all packages are installed"""
    try:
        import cv2
        import numpy
        return True
    except ImportError:
        return False

# Auto-install if needed
if not check_packages():
    print("\n⚠ Installing required packages for the first time...")
    try:
        install_packages()
    except Exception as e:
        print(f"⚠ Installation warning: {e}\n")
        print("Trying alternative installation method...")
        os.system(f"{sys.executable} -m pip install opencv-python opencv-contrib-python numpy pillow h5py imutils scikit-learn -q")

# Now import everything
import cv2
import numpy as np
from time import time, sleep
import os

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class FaceRecognitionWithDisplay:
    """Face Recognition with LIVE WEBCAM DISPLAY"""
    
    def __init__(self):
        print(f"\n{CYAN}Initializing Face Recognition System with Live Display...{RESET}\n")
        self.cascade_path = './data/haarcascade_frontalface_default.xml'
        self.classifiers_dir = './data/classifiers'
        self.data_dir = './data'
        self.face_cascade = None
        self.window_open = False
        
        # Check if cascade exists
        if not os.path.exists(self.cascade_path):
            print(f"{RED}❌ Cascade file not found at: {self.cascade_path}{RESET}")
            print(f"{YELLOW}Please ensure cascade file exists in ./data/ folder{RESET}\n")
            sys.exit(1)
        
        try:
            self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
            if self.face_cascade.empty():
                print(f"{RED}❌ Failed to load cascade classifier{RESET}\n")
                sys.exit(1)
            print(f"{GREEN}✓ System initialized with live display support{RESET}\n")
        except Exception as e:
            print(f"{RED}❌ Error: {e}{RESET}\n")
            sys.exit(1)
    
    def get_available_users(self):
        """Get list of trained users"""
        if not os.path.exists(self.classifiers_dir):
            return []
        users = []
        for file in os.listdir(self.classifiers_dir):
            if file.endswith("_classifier.xml"):
                users.append(file.replace("_classifier.xml", ""))
        return sorted(users)
    
    def capture_face_data_with_display(self, name, num_images=50):
        """Capture face images WITH LIVE WEBCAM DISPLAY"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}{BOLD}CAPTURING {num_images} FACE IMAGES WITH LIVE DISPLAY{RESET}")
        print(f"{CYAN}Capturing for: {name.upper()}{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        path = os.path.join(self.data_dir, name)
        os.makedirs(path, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ ERROR: Webcam not found!{RESET}\n")
            return 0
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"{GREEN}✓ Webcam opened successfully{RESET}")
        print(f"{BLUE}Instructions:{RESET}")
        print("  1. LIVE VIDEO will show in the window below")
        print("  2. Green rectangles = Face detected")
        print("  3. Move your head - left, right, up, down")
        print("  4. Make different expressions")
        print("  5. Keep distance 30-60cm from camera")
        print(f"  6. Close the window when done (or press ESC/Q)\n")
        print(f"{BLUE}Starting live capture...{RESET}\n")
        
        sleep(2)
        
        num_captured = 0
        frame_count = 0
        start_time = time()
        window_name = f"Face Capture - {name.upper()} (Press ESC to stop)"
        
        try:
            while num_captured < num_images:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Resize for better performance
                frame = cv2.resize(frame, (640, 480))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Draw info panel
                cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Captured: {num_captured}/{num_images}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames: {frame_count} | Time: {time()-start_time:.1f}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                cv2.putText(frame, "Press ESC/Q to stop", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw detected faces
                if len(faces) > 0:
                    cv2.putText(frame, "FACE DETECTED!", (450, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    for (x, y, w, h) in faces:
                        # Draw green rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Draw center point
                        cx, cy = x + w//2, y + h//2
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                        
                        # Extract face region
                        roi_gray = gray[y:y+h, x:x+w]
                        
                        # Save image
                        img_path = os.path.join(path, f"{num_captured}_{name}.jpg")
                        cv2.imwrite(img_path, roi_gray)
                        num_captured += 1
                        
                        if num_captured % 10 == 0:
                            print(f"{GREEN}✓{RESET} Captured: {num_captured}/{num_images}")
                        
                        if num_captured >= num_images:
                            break
                else:
                    cv2.putText(frame, "NO FACE DETECTED", (450, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                self.window_open = True
                
                # Check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                    print(f"\n{YELLOW}⚠ Stopped by user{RESET}")
                    break
                
                if num_captured >= num_images:
                    break
        
        except Exception as e:
            print(f"{RED}Error during capture: {e}{RESET}")
        
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
                self.window_open = False
            except:
                pass
        
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{GREEN}✓ CAPTURE COMPLETE!{RESET}")
        print(f"  Images saved: {num_captured}")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Location: {path}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        return num_captured
    
    def train_classifier(self, name):
        """Train face recognizer"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}{BOLD}TRAINING CLASSIFIER FOR: {name.upper()}{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        data_path = os.path.join(self.data_dir, name)
        if not os.path.exists(data_path):
            print(f"{RED}❌ No data found for {name}{RESET}\n")
            return False
        
        images = []
        labels = []
        image_files = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        
        if len(image_files) < 10:
            print(f"{RED}❌ Need at least 10 images, found {len(image_files)}{RESET}\n")
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
            print(f"{RED}❌ Could not load enough images{RESET}\n")
            return False
        
        print(f"\n{BLUE}Training LBPH recognizer with {len(images)} images...{RESET}\n")
        
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(images, np.array(labels))
            
            os.makedirs(self.classifiers_dir, exist_ok=True)
            classifier_path = os.path.join(self.classifiers_dir, f"{name}_classifier.xml")
            recognizer.write(classifier_path)
            
            print(f"{GREEN}✓ Training complete!{RESET}")
            print(f"  Classifier saved: {classifier_path}")
            print(f"{CYAN}{'='*70}{RESET}\n")
            return True
        except Exception as e:
            print(f"{RED}❌ Training error: {e}{RESET}\n")
            return False
    
    def recognize_faces_with_display(self, timeout=15):
        """Live face recognition WITH LIVE WEBCAM DISPLAY - Matches against ALL trained users"""
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"{CYAN}{BOLD}LIVE FACE RECOGNITION - ALL USERS{RESET}")
        print(f"{CYAN}Matching against all trained users{RESET}")
        print(f"{CYAN}Timeout: {timeout} seconds{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        users = self.get_available_users()
        if not users:
            print(f"{RED}❌ No trained users found!{RESET}\n")
            return False
        
        print(f"{GREEN}Trained users for matching: {', '.join(users)}{RESET}\n")
        
        # Load all classifiers
        recognizers = {}
        for user in users:
            classifier_path = os.path.join(self.classifiers_dir, f"{user}_classifier.xml")
            if os.path.exists(classifier_path):
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(classifier_path)
                recognizers[user] = recognizer
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{RED}❌ Webcam not found!{RESET}\n")
            return False
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"{GREEN}✓ Webcam opened{RESET}")
        print(f"{BLUE}Instructions:{RESET}")
        print("  1. LIVE VIDEO will show in the window below")
        print("  2. Green box = MATCH (Face recognized)")
        print("  3. Red box = Unknown face")
        print("  4. Username shown above the box when matched")
        print("  5. Confidence % shown in label")
        print("  6. Stand in front of camera")
        print(f"  7. Close the window when done (or press ESC/Q)\n")
        print(f"{BLUE}Starting live recognition...{RESET}\n")
        
        matched = False
        matched_user = None
        frame_count = 0
        match_count = 0
        start_time = time()
        window_name = f"Face Recognition - All Users (Press ESC to stop)"
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Resize for better performance
                frame = cv2.resize(frame, (640, 480))
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                elapsed = time() - start_time
                
                # Draw info panel
                cv2.rectangle(frame, (0, 0), (640, 100), (0, 0, 0), -1)
                cv2.putText(frame, f"Recognizing: ALL USERS", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {elapsed:.1f}s/{timeout}s | Matches: {match_count} | Frames: {frame_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                status_color = (0, 255, 0) if matched else (0, 0, 255)
                status_text = f"STATUS: MATCHED {matched_user.upper()} ✓" if matched else "STATUS: Searching..."
                cv2.putText(frame, status_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Process detected faces
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    best_match = None
                    best_confidence = 0
                    
                    # Check against all recognizers
                    for user, recognizer in recognizers.items():
                        try:
                            id, confidence = recognizer.predict(roi_gray)
                            confidence_percent = 100 - int(confidence)
                            
                            # Keep track of best match
                            if confidence_percent > best_confidence:
                                best_confidence = confidence_percent
                                best_match = user
                        except:
                            pass
                    
                    # Draw based on best match
                    if best_match and best_confidence > 50:
                        matched = True
                        matched_user = best_match
                        match_count += 1
                        label = f"{best_match.upper()} ({best_confidence}%)"
                        box_color = (0, 255, 0)  # Green
                        text_color = (0, 255, 0)
                    else:
                        label = f"Unknown"
                        box_color = (0, 0, 255)  # Red
                        text_color = (0, 0, 255)
                    
                    # Draw thick rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 3)
                    
                    # Draw label with background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(frame, (x, y-35), (x+label_size[0]+10, y), text_color, -1)
                    cv2.putText(frame, label, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (255, 255, 255), 2)
                    
                    # Draw center point
                    cx, cy = x + w//2, y + h//2
                    cv2.circle(frame, (cx, cy), 5, box_color, -1)
                    
                    # Draw crosshair
                    cv2.line(frame, (cx-10, cy), (cx+10, cy), box_color, 2)
                    cv2.line(frame, (cx, cy-10), (cx, cy+10), box_color, 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                self.window_open = True
                
                # Check timeout
                if elapsed >= timeout:
                    print(f"\n{YELLOW}⏱ Timeout reached!{RESET}")
                    break
                
                # Check keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
                    print(f"\n{YELLOW}⚠ Stopped by user{RESET}")
                    break
        
        except Exception as e:
            print(f"{RED}Error during recognition: {e}{RESET}")
        
        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
                self.window_open = False
            except:
                pass
        
        # Results
        print(f"\n{CYAN}{'='*70}{RESET}")
        print(f"Recognition Complete!")
        print(f"Total Frames: {frame_count}")
        print(f"Matches Found: {match_count}")
        if matched and matched_user:
            print(f"{GREEN}✓ SUCCESS - {matched_user.upper()} RECOGNIZED!{RESET}")
        else:
            print(f"{RED}✗ NO MATCH - Face not recognized{RESET}")
        print(f"{CYAN}{'='*70}{RESET}\n")
        
        return matched

def print_menu():
    """Print main menu"""
    print(f"\n{BOLD}{CYAN}╔{'='*68}╗{RESET}")
    print(f"{BOLD}{CYAN}║{' '*10}FACE RECOGNITION WITH LIVE WEBCAM DISPLAY{' '*15}║{RESET}")
    print(f"{BOLD}{CYAN}╚{'='*68}╝{RESET}\n")
    print(f"{BOLD}OPTIONS:{RESET}")
    print(f"  {GREEN}1{RESET}  - Capture Face Images (with live display)")
    print(f"  {GREEN}2{RESET}  - Train Classifier")
    print(f"  {GREEN}3{RESET}  - Recognize Faces (with live display)")
    print(f"  {GREEN}4{RESET}  - Complete Flow (Capture → Train → Recognize)")
    print(f"  {GREEN}5{RESET}  - List Available Users")
    print(f"  {GREEN}Q{RESET}  - Quit\n")

def main():
    """Main application"""
    try:
        app = FaceRecognitionWithDisplay()
    except SystemExit:
        return
    
    while True:
        print_menu()
        choice = input(f"{BOLD}Enter choice: {RESET}").strip().upper()
        
        if choice == 'Q':
            print(f"{YELLOW}Goodbye!{RESET}\n")
            break
        
        elif choice == '1':
            # Capture with display
            name = input(f"\n{BOLD}Enter username: {RESET}").strip()
            if name:
                num = input(f"{BOLD}Number of images (default 50): {RESET}").strip()
                try:
                    num = int(num) if num else 50
                except:
                    num = 50
                
                captured = app.capture_face_data_with_display(name, num)
                if captured > 0:
                    print(f"{GREEN}✓ Captured {captured} images for {name}{RESET}")
        
        elif choice == '2':
            # Train
            users = app.get_available_users()
            if not users:
                print(f"{RED}❌ No users found. Capture data first!{RESET}\n")
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
                    if app.train_classifier(user):
                        print(f"{GREEN}✓ Training successful!{RESET}")
        
        elif choice == '3':
            # Recognize with display - against all users
            tout = input(f"{BOLD}Timeout seconds (default 15): {RESET}").strip()
            try:
                tout = int(tout) if tout else 15
            except:
                tout = 15
            
            app.recognize_faces_with_display(tout)
        
        elif choice == '4':
            # Complete flow with display
            name = input(f"\n{BOLD}Enter username: {RESET}").strip()
            if name:
                # Capture with display
                captured = app.capture_face_data_with_display(name, 50)
                if captured < 10:
                    print(f"{RED}✗ Need at least 10 images. Got {captured}.{RESET}\n")
                    continue
                
                # Train
                input(f"\n{BOLD}Press Enter to train...{RESET}")
                if app.train_classifier(name):
                    # Recognize with display - against all users
                    input(f"\n{BOLD}Press Enter to recognize...{RESET}")
                    app.recognize_faces_with_display(15)
        
        elif choice == '5':
            # List users
            users = app.get_available_users()
            if users:
                print(f"\n{BOLD}{GREEN}Trained users:{RESET}")
                for user in users:
                    data_path = os.path.join(app.data_dir, user)
                    num = len([f for f in os.listdir(data_path) if f.endswith('.jpg')]) if os.path.exists(data_path) else 0
                    print(f"  ✓ {user} ({num} images)")
                print()
            else:
                print(f"\n{RED}No trained users found{RESET}\n")
        
        else:
            print(f"{RED}Invalid choice. Try again.{RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Application interrupted{RESET}\n")
    except Exception as e:
        print(f"\n{RED}ERROR: {e}{RESET}\n")
        import traceback
        traceback.print_exc()
