#!/usr/bin/env python3
"""
ULTRA-PRECISION FACE RECOGNITION - FIXED FOR WINDOWS
No unicode, no f-strings - pure Windows compatible
"""

import subprocess
import sys
import os
import cv2
import numpy as np
from time import time, sleep
from collections import deque

def ensure_packages():
    packages = ["opencv-python", "opencv-contrib-python", "numpy", "pillow"]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except:
            print("[INSTALL] {}...".format(pkg))
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

ensure_packages()

class FaceRecognitionSystem:
    def __init__(self):
        print("\n" + "="*60)
        print("FACE RECOGNITION - ULTRA PRECISION MODE")
        print("="*60 + "\n")
        
        self.cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        if self.cascade.empty():
            print("[ERROR] Cascade not found!")
            sys.exit(1)
        
        self.recognizers = {}
        self.load_recognizers()
        print("[OK] System ready!\n")
    
    def load_recognizers(self):
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
                    print("[+] Loaded: {}".format(name))
                except:
                    pass
    
    def preprocess_face(self, face_roi):
        try:
            # Histogram equalization
            face_eq = cv2.equalizeHist(face_roi)
            
            # CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            face_clahe = clahe.apply(face_eq)
            
            # Bilateral filter
            face_filtered = cv2.bilateralFilter(face_clahe, 9, 75, 75)
            
            # Morphological
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            face_morph = cv2.morphologyEx(face_filtered, cv2.MORPH_CLOSE, kernel)
            
            # Sharpen
            kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 1.0
            face_sharp = cv2.filter2D(face_morph, -1, kernel_sharp)
            
            face_sharp = np.clip(face_sharp, 0, 255).astype(np.uint8)
            return face_sharp
        except:
            return face_roi
    
    def capture_faces(self, name, num=100):
        print("\n" + "="*60)
        print("CAPTURE - {} (100 IMAGES)".format(name.upper()))
        print("="*60 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera!")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        out_dir = './data/{}'.format(name)
        os.makedirs(out_dir, exist_ok=True)
        
        count = 0
        start = time()
        window = "CAPTURE - {}".format(name.upper())
        
        print("[OK] Camera opened - Stand in front of camera")
        print("[*] Capturing {} images...\n".format(num))
        
        while count < num:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.cascade.detectMultiScale(gray, 1.05, 7, minSize=(100, 100), maxSize=(400, 400))
            
            h, w = frame.shape[:2]
            
            # Add overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 40), -1)
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            
            # Title
            cv2.putText(frame, "CAPTURING - {}".format(name.upper()), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            
            # Progress
            pct = int((count / num) * 100)
            bar_w = int((w - 80) * count / num)
            cv2.rectangle(frame, (40, 140), (w-40, 180), (100, 100, 100), 2)
            cv2.rectangle(frame, (40, 140), (40+bar_w, 180), (0, 255, 0), -1)
            cv2.putText(frame, "Progress: {}/{} ({}%)".format(count, num, pct), (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            elapsed = time() - start
            cv2.putText(frame, "Time: {:.1f}s".format(elapsed), (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            
            if len(faces) > 0:
                cv2.putText(frame, "Face detected - CAPTURING", (50, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Move closer to camera", (50, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
            
            # Capture
            for (x, y, ww, hh) in faces:
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 5)
                cv2.circle(frame, (x + ww//2, y + hh//2), 5, (0, 255, 0), -1)
                
                face_roi = gray[y:y+hh, x:x+ww]
                face_preprocessed = self.preprocess_face(face_roi)
                
                cv2.imwrite('{}/{}_{}jpg'.format(out_dir, count, name), face_preprocessed, [cv2.IMWRITE_JPEG_QUALITY, 99])
                
                count += 1
                print("[+] Captured: {}/{}".format(count, num))
                
                if count >= num:
                    break
            
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("\n[*] Stopped by user\n")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        
        if count > 0:
            print("\n" + "="*60)
            print("[OK] CAPTURE COMPLETE!")
            print("[OK] Images: {}".format(count))
            print("[OK] Quality: ULTRA-HIGH")
            print("="*60 + "\n")
            return True
        return False
    
    def train_classifier(self, name):
        print("\n" + "="*60)
        print("TRAIN - {} (ENHANCED LBPH)".format(name.upper()))
        print("="*60 + "\n")
        
        data_dir = './data/{}'.format(name)
        if not os.path.exists(data_dir):
            print("[ERROR] No images found\n")
            return False
        
        images = []
        labels = []
        
        print("[*] Loading images...")
        for img_name in sorted(os.listdir(data_dir)):
            if img_name.endswith('.jpg'):
                img = cv2.imread(os.path.join(data_dir, img_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = self.preprocess_face(img)
                    images.append(img)
                    labels.append(0)
        
        if len(images) < 2:
            print("[ERROR] Need at least 2 images\n")
            return False
        
        print("[*] Training {} images...".format(len(images)))
        
        rec = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=16, grid_y=16)
        rec.train(images, np.array(labels))
        
        os.makedirs('./data/classifiers', exist_ok=True)
        rec.save('./data/classifiers/{}_classifier.xml'.format(name))
        
        self.load_recognizers()
        
        print("\n" + "="*60)
        print("[OK] TRAINING COMPLETE!")
        print("[OK] Images: {}".format(len(images)))
        print("[OK] Model: LBPH (Enhanced)")
        print("="*60 + "\n")
        return True
    
    def recognize_all(self, timeout=15):
        print("\n" + "="*60)
        print("RECOGNIZE - AUTO-DETECT ALL USERS")
        print("Timeout: {}s".format(timeout))
        print("="*60 + "\n")
        
        if not self.recognizers:
            print("[ERROR] No trained users!\n")
            return
        
        print("[*] Users: {}".format(', '.join(self.recognizers.keys())))
        print("[OK] Opening camera...\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera!\n")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        start = time()
        matches = {name: 0 for name in self.recognizers.keys()}
        frame_count = 0
        window = "RECOGNITION - AUTO-DETECT"
        
        print("[OK] Camera opened - Stand in front\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.cascade.detectMultiScale(gray, 1.05, 7, minSize=(100, 100), maxSize=(400, 400))
            
            elapsed = time() - start
            frame_count += 1
            h, w = frame.shape[:2]
            
            # Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 40), -1)
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            
            # Title
            cv2.putText(frame, "AUTO-DETECT ALL USERS", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
            cv2.putText(frame, "Time: {:.1f}s / {}s".format(elapsed, timeout), (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)
            cv2.putText(frame, "Frames: {}".format(frame_count), (40, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 2)
            
            # Process faces
            for (x, y, ww, hh) in faces:
                face_roi = gray[y:y+hh, x:x+ww]
                face_preprocessed = self.preprocess_face(face_roi)
                
                best_name = None
                best_conf = 100
                
                for name, rec in self.recognizers.items():
                    label, conf = rec.predict(face_preprocessed)
                    if conf < best_conf:
                        best_conf = conf
                        best_name = name
                
                if best_name and best_conf < 50:
                    color = (0, 255, 0)
                    text = "{} ({:.0f}%)".format(best_name.upper(), best_conf)
                    matches[best_name] += 1
                    border = 5
                else:
                    color = (0, 0, 255)
                    text = "UNKNOWN ({:.0f}%)".format(best_conf)
                    border = 3
                
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), color, border)
                cv2.rectangle(frame, (x-2, y-50), (x+len(text)*12+5, y-10), color, -1)
                cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
                cv2.circle(frame, (x + ww//2, y + hh//2), 5, color, -1)
            
            # Summary
            match_text = []
            for name, count in matches.items():
                if count > 0:
                    match_text.append("{}:{}".format(name, count))
            
            if match_text:
                cv2.putText(frame, "MATCHES: " + " | ".join(match_text), (40, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            
            cv2.imshow(window, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or elapsed >= timeout:
                if key == 27:
                    print("\n[*] Stopped by user")
                else:
                    print("\n[*] Timeout reached")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        sleep(0.5)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print("Frames: {}, Duration: {:.1f}s\n".format(frame_count, elapsed))
        
        total = sum(matches.values())
        for name, count in matches.items():
            if count > 0:
                print("[OK] {}: {} matches".format(name.upper(), count))
            else:
                print("     {}: 0 matches".format(name.upper()))
        
        if total > 0:
            print("\n[OK] RECOGNIZED {} FACES!\n".format(total))
        else:
            print("\n[!] NO MATCHES FOUND\n")
        
        print("="*60 + "\n")
    
    def list_users(self):
        print("\n" + "="*60)
        print("TRAINED USERS")
        print("="*60 + "\n")
        
        classifiers = []
        if os.path.exists('./data/classifiers'):
            classifiers = [f.replace('_classifier.xml', '') for f in os.listdir('./data/classifiers') if f.endswith('_classifier.xml')]
        
        if not classifiers:
            print("[*] No users trained\n")
            return
        
        print("Total: {}\n".format(len(classifiers)))
        
        for i, user in enumerate(sorted(classifiers), 1):
            user_dir = './data/{}'.format(user)
            img_count = len([f for f in os.listdir(user_dir) if f.endswith('.jpg')]) if os.path.exists(user_dir) else 0
            print("  {}. {} ({} images)".format(i, user.upper(), img_count))
        
        print()
    
    def complete_flow(self, name):
        print("\n" + "="*60)
        print("COMPLETE FLOW - CAPTURE + TRAIN + RECOGNIZE")
        print("="*60 + "\n")
        
        if not self.capture_faces(name, 100):
            return
        
        if not self.train_classifier(name):
            return
        
        raw_input("[*] Press Enter to recognize...")
        self.recognize_all(15)
    
    def menu(self):
        while True:
            print("\n" + "="*60)
            print("FACE RECOGNITION - ULTRA PRECISION")
            print("="*60)
            print("\nMENU:")
            print("  1 - Capture (100 images)")
            print("  2 - Train (Enhanced)")
            print("  3 - Recognize (Custom timeout)")
            print("  4 - Complete Flow")
            print("  5 - List Users")
            print("  Q - Quit\n")
            
            choice = raw_input("Choice: ").strip().upper()
            
            if choice == '1':
                name = raw_input("Name: ").strip()
                if name:
                    self.capture_faces(name, 100)
            
            elif choice == '2':
                name = raw_input("Name: ").strip()
                if name:
                    self.train_classifier(name)
            
            elif choice == '3':
                print("[*] Timeout: 1=10s, 2=15s, 3=30s, 4=60s, 5=custom")
                t = raw_input("Choice: ").strip()
                timeout = {'1': 10, '2': 15, '3': 30, '4': 60}.get(t, 15)
                if t == '5':
                    try:
                        timeout = int(raw_input("Seconds: "))
                    except:
                        pass
                self.recognize_all(timeout)
            
            elif choice == '4':
                name = raw_input("Name: ").strip()
                if name:
                    self.complete_flow(name)
            
            elif choice == '5':
                self.list_users()
            
            elif choice == 'Q':
                print("[OK] Goodbye!\n")
                break

if __name__ == "__main__":
    try:
        app = FaceRecognitionSystem()
        app.menu()
    except KeyboardInterrupt:
        print("\n[*] Interrupted\n")
    except Exception as e:
        print("\n[ERROR] {}\n".format(e))
        import traceback
        traceback.print_exc()
