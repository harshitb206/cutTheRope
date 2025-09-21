
# Cut the Rope - Hand Controlled (Mini Game)

**Description**
A lightweight "Cut the Rope" style game controlled by hand gestures using MediaPipe + OpenCV.
Cut swinging ropes with your index fingertip to drop the ball into a moving basket.
This project includes sounds, obstacles, lives, and scoring. It's minimal-GUI and uses webcam input.

**How to run**
1. (Optional) Create and activate a virtual environment
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python main.py
   ```

**Controls / Notes**
- Use your index fingertip (pointing with your index) to slice ropes in front of the camera.
- Press ESC to quit.
- The project generates two short WAV files (cut.wav and catch.wav) used for sound effects.
- Tested on Python 3.8+. If pygame fails to initialize audio, try installing SDL dependencies for your OS.

**Files**
- main.py -- game code
- requirements.txt -- python deps
- cut.wav, catch.wav -- generated sound effects
- README.md -- this file

Enjoy! 🎮


**New Feature:** High scores are automatically saved in `highscore.txt`.


**New Feature:** Start menu and Game Over screen controlled by hand gestures (hover + pinch). Buttons are fully OpenCV-drawn.
