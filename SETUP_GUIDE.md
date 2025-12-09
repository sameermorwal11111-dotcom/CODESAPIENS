# Face Recognition GUI App - Setup & Running Guide

## Quick Start Commands

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Option A - Run GUI Application (Full Features)
```powershell
python app-gui.py
```

### Step 2: Option B - Run Webcam Recognition (Direct Recognition)
```powershell
python run_webcam_recognition.py
```

---

## What's Updated

### Updated requirements.txt
- ✓ opencv-python: 4.8.0.76 (latest stable)
- ✓ opencv-contrib-python: 4.8.0.76 (with extra modules)
- ✓ Pillow: 10.0.0 (image processing)
- ✓ h5py: 3.9.0 (deep learning models)
- ✓ imutils: 0.5.4 (image utilities)
- ✓ keras: 2.13.1 (neural networks)
- ✓ tensorflow: 2.13.0 (deep learning)
- ✓ numpy: 1.24.3 (numerical computing)
- ✓ scikit-learn: 1.3.0 (machine learning)

---

## Usage Guide

### Using GUI Application (Recommended)

```powershell
python app-gui.py
```

**Steps:**
1. **Sign Up** - Enter a new username
2. **Capture Data** - Stand in front of webcam, let it capture 300 images
3. **Train Model** - Click "Train The Model" to create classifier
4. **Recognition** - Click "Face Recognition" to test

### Using Direct Webcam Recognition (Simple)

```powershell
python run_webcam_recognition.py
```

**Features:**
- Lists all available trained users
- Select user to recognize
- Runs for 10 seconds (auto-timeout)
- Press 'Q' to exit early
- Shows confidence percentage
- Green box = Recognized, Red box = Unknown

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit/Exit recognition |
| **ESC** | Exit (in capture mode) |

---

## Troubleshooting

### Issue: "No trained classifiers found"
**Solution:** 
1. Run `python app-gui.py`
2. Create a new user and train a model first
3. Then use `python run_webcam_recognition.py`

### Issue: Webcam not opening
**Solution:**
1. Check if another application is using webcam
2. Restart VS Code
3. Run: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Issue: Import errors
**Solution:**
```powershell
pip install -r requirements.txt --upgrade
```

### Issue: Face not detected
**Solution:**
- Ensure good lighting
- Face should be clear and frontal
- Distance: 30-60cm from camera
- Remove glasses/sunglasses if possible

---

## File Structure

```
data/
  ├── haarcascade_frontalface_default.xml     (Face detection model)
  ├── classifiers/
  │   ├── username1_classifier.xml            (Trained models)
  │   ├── username2_classifier.xml
  │   └── ...
  ├── username1/                               (Training data)
  │   ├── 0username1.jpg
  │   ├── 1username1.jpg
  │   └── ... (300 images)
  └── ...

app-gui.py                                    (Main GUI application)
run_webcam_recognition.py                    (Direct webcam recognition)
requirements.txt                              (Python dependencies)
nameslist.txt                                (User list)
```

---

## Performance Tips

1. **Training**: Use at least 300 images per user
2. **Lighting**: Uniform lighting works best
3. **Angles**: Capture faces from different angles
4. **Distance**: Maintain consistent distance during capture
5. **Expression**: Vary expressions (smile, neutral, etc.)

---

## Quick Command Reference

```powershell
# Install dependencies
pip install -r requirements.txt

# Run GUI App
python app-gui.py

# Run Webcam Recognition
python run_webcam_recognition.py

# Check webcam status
python -c "import cv2; cap = cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'Webcam ERROR'); cap.release()"

# Check Python version
python --version

# Check installed packages
pip list | grep opencv
```

---

## System Requirements

- **Python**: 3.7 or higher
- **Webcam**: USB or built-in camera
- **RAM**: 2GB minimum, 4GB+ recommended
- **Disk**: 500MB free space
- **OS**: Windows 10/11, macOS, Linux

---

## Support

If you encounter issues:
1. Check above troubleshooting section
2. Verify all requirements are installed
3. Ensure webcam permissions are granted
4. Test webcam with another app first

Enjoy your Face Recognition System! 🎉
