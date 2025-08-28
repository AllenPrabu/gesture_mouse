# Gesture OS

Gesture OS is a Python-based application that lets you control your mouse, switch applications, scroll, click, and even type or execute keyboard shortcuts using just your hand gestures and voice commands. It uses your webcam for hand tracking and your microphone for speech recognition, providing a futuristic, touchless way to interact with your computer—just like in sci-fi movies!

## Features

- **Hand Gesture Mouse Control:** Move your mouse cursor by pointing your finger.
- **Clicking:** Touch your index finger and thumb to left-click, or your middle finger and thumb to right-click.
- **Scrolling:** Raise three fingers (index, middle, ring) and lower your pinky to scroll up or down.
- **App Switching (Alt+Tab):** Make a fist to open the app switcher, then move your fist left/right to change tabs, and release to select.
- **Voice Commands:** Say "computer" or "jarvis" to activate, then speak commands like "press enter", "backspace", "copy", "paste", or dictate text to type automatically.

## Requirements

- Python 3.8+
- Webcam and microphone
- Windows OS


## Installation (To edit)

1. **Clone or download this repository.**
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   Or manually:
   ```
   pip install opencv-python mediapipe mouse keyboard SpeechRecognition pyaudio numpy
   ```

3. **(Optional) If you have issues with PyAudio:**
   ```
   pip install pipwin
   pipwin install pyaudio
   ```

## How to Run

1. **Plug in your webcam and microphone.**
2. **Open a terminal in the project folder.**
3. **Run:**
   ```
   python main.py
   ```
4. **A window will open showing your webcam feed.**
5. **Use hand gestures and voice commands as described above!**
6. **Press `q` in the webcam window to quit.**

## Download the EXE

If you don't want to install Python or dependencies, you can simply download the pre-built Windows executable:

- [Download Gesture OS .exe](https://drive.google.com/file/d/1erzvOf5C6yWHpn07CxnKr8Fg5fqgUy_P/view?usp=sharing)  

Just download and run the `.exe` file. No installation required!

## Packaging as an EXE (for developers)

If you want to create a standalone Windows executable yourself:
```
pyinstaller --onefile --noconsole main.py
```
The `.exe` will appear in the `dist` folder.

## Notes

- For best results, use in a well-lit environment.
- You can change the wake word ("computer" or "jarvis") in the code.
- All configuration options (like scroll speed) are in `config.py`.

---

Enjoy your futuristic gesture and voice-controlled OS