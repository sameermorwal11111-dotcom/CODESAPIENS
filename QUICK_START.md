# DIRECT COPY-PASTE COMMANDS TO RUN

## ONE-COMMAND SETUP & RUN

### Step 1: Install Dependencies (Run Once)
```powershell
pip install -r requirements.txt
```

### Step 2: Run the Complete Face Recognition System
```powershell
python face_recognition_complete.py
```

---

## WHAT YOU GET

This is a **COMPLETE ALL-IN-ONE** application with:

✅ **Add Face Data** - Capture new faces from webcam  
✅ **Train Classifier** - Automatically train the model  
✅ **Live Recognition** - Recognize faces in real-time with name display  
✅ **Complete Flow** - All 3 steps in one go  
✅ **View Users** - See all trained users and their data  

---

## MENU OPTIONS EXPLAINED

```
╔════════════════════════════════════════════════════════╗
║        FACE RECOGNITION COMPLETE SYSTEM               ║
╚════════════════════════════════════════════════════════╝

Options:
  1  - Add New Face Data (Capture)
  2  - Train Classifier
  3  - Live Face Recognition
  4  - Complete Flow (Capture + Train + Recognize)
  5  - View Available Users
  Q  - Quit
```

### Option 1: Add Face Data
- Enter username
- Choose number of images (default 50)
- Stand in front of webcam
- System captures faces automatically
- Press Q/ESC to stop

### Option 2: Train Classifier
- Select user from list
- System learns from captured images
- Creates a trained classifier file

### Option 3: Live Face Recognition
- Select user to recognize
- Webcam opens and compares faces
- Shows "MATCH" or "Unknown" with confidence
- Name displays in GREEN for match
- Timeout: 15 seconds (configurable)

### Option 4: Complete Flow
- All-in-one: Capture → Train → Recognize
- Perfect for new users
- Most convenient option!

### Option 5: View Users
- Lists all trained users
- Shows number of training images per user

---

## QUICK START EXAMPLE

### First Time: Complete Setup

```powershell
# 1. Install (first time only)
pip install -r requirements.txt

# 2. Run the system
python face_recognition_complete.py

# 3. Follow prompts:
# Menu: Select option 4 (Complete Flow)
# Enter: ngoc
# Capture: Stand in front of webcam
# Train: Wait for training to complete
# Recognize: Get instant feedback with your name!
```

### Later: Just Recognize

```powershell
python face_recognition_complete.py

# Menu: Select option 3 (Live Face Recognition)
# Select: ngoc (or just type "1")
# Get instant results with name display!
```

---

## KEYBOARD CONTROLS

| Key | Action |
|-----|--------|
| Q | Quit webcam / Exit menu |
| ESC | Stop capturing |
| Enter | Continue / Confirm |

---

## EXAMPLE WORKFLOW

**First Time (Add yourself):**
```
python face_recognition_complete.py
→ Enter choice: 4
→ Enter username: john
→ Captures 50 images
→ Trains classifier
→ Live recognition (Shows "MATCH: JOHN")
```

**Later (Recognize anyone):**
```
python face_recognition_complete.py
→ Enter choice: 3
→ Select user: 1 (or type "john")
→ Webcam shows live name recognition
```

---

## FILE LOCATIONS

Trained models saved here:
```
data/classifiers/
  ├── ngoc_classifier.xml
  ├── ab_classifier.xml
  ├── john_classifier.xml
  └── ...

Face images saved here:
data/
  ├── ngoc/
  │   ├── 0_ngoc.jpg
  │   ├── 1_ngoc.jpg
  │   └── ...
  ├── ab/
  │   └── ...
  └── john/
      └── ...
```

---

## FEATURES INCLUDED

🎥 **Real-time Webcam**
- Live preview while capturing
- Face detection with rectangles
- Progress counter

📊 **Training**
- Automatic LBPH algorithm
- Minimum 10 images required
- Saves trained model

🎯 **Recognition**
- Live name display
- Confidence percentage
- Color-coded feedback (Green = Match, Red = Unknown)
- Configurable timeout

🎨 **User Interface**
- Colored terminal output
- Clear instructions
- Menu-driven interface
- Error handling

---

## TROUBLESHOOTING

**"No users found"**
```
→ Need to capture face data first
→ Select option 1 or 4 to add a user
```

**Webcam not opening**
```
→ Close other apps using webcam
→ Restart VS Code
→ Try: python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Face not detected**
```
→ Better lighting
→ Clear frontal face
→ 30-60cm from camera
→ Remove glasses/sunglasses
```

**Training failed**
```
→ Need at least 10 images
→ Try capturing more images
→ Ensure images are saved correctly
```

---

## RECOMMENDED STEPS

1. Run `python face_recognition_complete.py`
2. Choose option 4 (Complete Flow)
3. Enter your name
4. Wait for webcam to open
5. Let it capture 50 images (move around)
6. System automatically trains
7. See real-time recognition with your name!

That's it! 🎉

---

**Ready?** Just run:
```powershell
python face_recognition_complete.py
```
