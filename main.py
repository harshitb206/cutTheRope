import cv2
import mediapipe as mp
import numpy as np
import random, math, os, time
import pygame

# --- Setup paths & sounds ---
try:
    pygame.mixer.init()
except Exception as e:
    print("Warning: pygame.mixer init failed:", e)

base_dir = os.path.dirname(__file__)
if base_dir == "":
    base_dir = "."

def load_sound(name):
    try:
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return pygame.mixer.Sound(path)
    except Exception as e:
        print(f"Sound load failed ({name}):", e)
        return None

cut_sound = load_sound("cut.wav")
catch_sound = load_sound("catch.wav")

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# --- Window and camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Optionally set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# --- Game constants ---
WIDTH = 640
HEIGHT = 480
ball_radius = 18
gravity = 3.0

# reduce CPU by processing hands every N frames
frame_skip = 2

# --- High score ---
highscore_file = os.path.join(base_dir, "highscore.txt")
try:
    with open(highscore_file, "r") as hf:
        high_score = int(hf.read().strip())
except:
    high_score = 0

# --- Helper functions ---
def pendulum_pos(anchor, length, ang):
    x = int(anchor[0] + length * math.sin(ang))
    y = int(anchor[1] + length * math.cos(ang))
    return x, y

def point_line_distance(px, py, x1, y1, x2, y2):
    num = abs((y2 - y1)*px - (x2 - x1)*py + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    if den == 0:
        return math.hypot(px - x1, py - y1)
    return num / den

def segment_intersect(a1, a2, b1, b2):
    # Robust segment intersection using orientations
    def orient(p, q, r):
        # cross product (q - p) x (r - p)
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        # check if point q lies on segment pr
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    p1, p2, p3, p4 = a1, a2, b1, b2
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)

    # General case
    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    # Special Cases: collinear and overlapping
    if o1 == 0 and on_seg(p1, p3, p2):
        return True
    if o2 == 0 and on_seg(p1, p4, p2):
        return True
    if o3 == 0 and on_seg(p3, p1, p4):
        return True
    if o4 == 0 and on_seg(p3, p2, p4):
        return True

    return False

def generate_obstacles(num):
    obs = []
    min_y = 180
    max_y = HEIGHT - 200
    if max_y <= min_y:
        min_y = 160
        max_y = HEIGHT - 160
    for _ in range(num):
        w = random.randint(40, 100)
        h = random.randint(10, 20)
        x = random.randint(20, WIDTH - w - 20)
        y = random.randint(min_y, max_y)
        obs.append((x,y,w,h))
    return obs

def draw_button(frame, rect, text, hovered=False):
    x,y,wid,ht = rect
    color = (0,200,255) if hovered else (180,180,180)
    thickness = -1 if hovered else 2
    cv2.rectangle(frame, (x,y), (x+wid, y+ht), color, thickness)
    cv2.putText(frame, text, (x + 10, y + ht//2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0) if hovered else (255,255,255), 2)

# Detect pinch between index tip (8) and thumb tip (4)
def is_pinch(landmarks, w, h, thresh=0.05):
    try:
        x1 = landmarks.landmark[8].x * w
        y1 = landmarks.landmark[8].y * h
        x2 = landmarks.landmark[4].x * w
        y2 = landmarks.landmark[4].y * h
        d = math.hypot(x2-x1, y2-y1)
        return d < (thresh * w)
    except Exception:
        return False

# --- Menus ---
def start_menu():
    start_rect = (WIDTH//2 - 120, HEIGHT//2 - 40, 240, 60)
    while True:
        ret, frame = cap.read()
        if not ret:
            return False
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        finger_x = None; finger_y = None; pinched = False
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            finger_x = int(lm.landmark[8].x * w)
            finger_y = int(lm.landmark[8].y * h)
            cv2.circle(frame, (finger_x, finger_y), 7, (0,255,0), -1)
            pinched = is_pinch(lm, w, h, thresh=0.045)

        cv2.putText(frame, "Cut the Rope - Hand Game", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(frame, "Hover index on START and pinch to begin", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, f"HighScore: {high_score}", (WIDTH-200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        hovered = False
        if finger_x is not None and finger_y is not None:
            x,y,wid,ht = start_rect
            if x < finger_x < x+wid and y < finger_y < y+ht:
                hovered = True
                cv2.circle(frame, (finger_x, finger_y), 12, (0,255,0), 2)
                if pinched:
                    time.sleep(0.25)
                    return True

        draw_button(frame, start_rect, "START", hovered)
        cv2.imshow("Cut the Rope", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False


def game_over_menu(final_score):
    restart_rect = (WIDTH//2 - 260, HEIGHT//2 + 40, 220, 60)
    quit_rect = (WIDTH//2 + 40, HEIGHT//2 + 40, 220, 60)
    global high_score
    while True:
        ret, frame = cap.read()
        if not ret:
            return "quit"
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        finger_x = None; finger_y = None; pinched = False
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            finger_x = int(lm.landmark[8].x * w)
            finger_y = int(lm.landmark[8].y * h)
            cv2.circle(frame, (finger_x, finger_y), 7, (0,255,0), -1)
            pinched = is_pinch(lm, w, h, thresh=0.045)

        cv2.putText(frame, "GAME OVER", (WIDTH//2 - 120, HEIGHT//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.putText(frame, f"Score: {final_score}", (WIDTH//2 - 90, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"HighScore: {high_score}", (WIDTH//2 - 90, HEIGHT//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        restart_hover = False; quit_hover = False
        if finger_x is not None and finger_y is not None:
            rx,ry,rw,rh = restart_rect
            qx,qy,qw,qh = quit_rect
            if rx < finger_x < rx+rw and ry < finger_y < ry+rh:
                restart_hover = True
                if pinched:
                    time.sleep(0.25)
                    return "restart"
            if qx < finger_x < qx+qw and qy < finger_y < qy+qh:
                quit_hover = True
                if pinched:
                    time.sleep(0.25)
                    return "quit"

        draw_button(frame, restart_rect, "RESTART", restart_hover)
        draw_button(frame, quit_rect, "QUIT", quit_hover)

        cv2.imshow("Cut the Rope", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return "quit"

# --- Core Game Loop ---
def game_loop():
    global high_score
    score = 0
    lives = 3
    level = 1

    rope_anchor = (WIDTH // 2, 60)
    angle = math.pi / 4
    angular_velocity = 0.02
    rope_length = 140
    cut = False
    fall_velocity = 0.0
    last_cut_time = 0

    basket_w, basket_h = 160, 60

    # --- FIXED: Basket spawns below rope anchor ---
    basket_x = max(60, min(WIDTH - basket_w - 60, rope_anchor[0] - basket_w // 2 + random.randint(-40, 40)))
    basket_y = HEIGHT - 120

    obstacles = generate_obstacles(1)
    ball_x, ball_y = pendulum_pos(rope_anchor, rope_length, angle)

    prev_finger = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            return "quit", score
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        h, w, _ = frame.shape

        finger_x = None; finger_y = None; pinched = False
        lm = None
        if frame_count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                finger_x = int(lm.landmark[8].x * w)
                finger_y = int(lm.landmark[8].y * h)
                cv2.circle(frame, (finger_x, finger_y), 7, (0,255,0), -1)
                pinched = is_pinch(lm, w, h, thresh=0.045)
        frame_count += 1

        for (ox,oy,ow,oh) in obstacles:
            cv2.rectangle(frame, (ox,oy), (ox+ow, oy+oh), (0,0,255), -1)

        bowl_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cx = basket_x + basket_w//2
        cy = basket_y + basket_h
        axes = (basket_w//2, basket_h)
        cv2.ellipse(bowl_mask, (cx, cy), axes, 0, 0, 180, 255, -1)
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 180, (0,255,200), 2)
        cv2.ellipse(frame, (cx, cy), (axes[0]-6, axes[1]-6), 0, 0, 180, (20,100,60), -1)

        cv2.putText(frame, f"Lives: {lives}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Score: {score}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Level: {level}", (10,85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(frame, f"HighScore: {high_score}", (WIDTH-240, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)

        if not cut:
            angle += angular_velocity * (1 + (level-1)*0.08)
            ball_x, ball_y = pendulum_pos(rope_anchor, rope_length, angle)
            cv2.line(frame, rope_anchor, (ball_x, ball_y), (255,0,0), 2)
        else:
            fall_velocity += gravity * 0.25
            ball_y += int(fall_velocity)
            ball_x += int(math.sin(angle) * 2)

        cv2.circle(frame, (ball_x, ball_y), ball_radius, (0,0,255), -1)
        cv2.circle(frame, rope_anchor, 4, (255,255,255), -1)

        # Cutting logic: require previous finger to exist
        if not cut and prev_finger is not None and finger_x is not None:
            dx = finger_x - prev_finger[0]
            dy = finger_y - prev_finger[1]
            move_dist = math.hypot(dx, dy)
            min_swipe = 12
            if move_dist > min_swipe:
                a1 = (prev_finger[0], prev_finger[1])
                a2 = (finger_x, finger_y)
                b1 = rope_anchor
                b2 = (ball_x, ball_y)
                if segment_intersect(a1, a2, b1, b2) and time.time() - last_cut_time > 0.2:
                    cut = True
                    fall_velocity = 0.0
                    last_cut_time = time.time()
                    if cut_sound:
                        cut_sound.play()

        # Update prev_finger when we have a detection
        if finger_x is not None and finger_y is not None:
            prev_finger = (finger_x, finger_y)

        cracked = False
        for (ox,oy,ow,oh) in obstacles:
            if (ox < ball_x < ox+ow) and (oy < ball_y+ball_radius < oy+oh):
                lives -= 1
                cracked = True
                break

        if cracked:
            cv2.putText(frame, "Ouch!", (WIDTH//2 - 40, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            cv2.imshow("Cut the Rope", frame)
            cv2.waitKey(200)
            rope_anchor = (random.randint(120, WIDTH-120), 60)
            angle = math.pi/4
            cut = False
            fall_velocity = 0.0
            ball_x, ball_y = pendulum_pos(rope_anchor, rope_length, angle)

            # --- FIXED basket spawn ---
            basket_x = max(60, min(WIDTH - basket_w - 60, rope_anchor[0] - basket_w // 2 + random.randint(-40, 40)))
            obstacles = generate_obstacles(max(1, level//2))
            if lives <= 0:
                return "gameover", score

        if cut:
            bx = int(ball_x); by = int(ball_y + ball_radius//2)
            if 0 <= bx < WIDTH and 0 <= by < HEIGHT:
                if bowl_mask[by, bx] > 0:
                    score += 1
                    if catch_sound:
                        catch_sound.play()
                    if score > high_score:
                        high_score = score
                        try:
                            with open(highscore_file, "w") as hf:
                                hf.write(str(high_score))
                        except:
                            pass
                    if score % 3 == 0:
                        level += 1
                    rope_anchor = (random.randint(120, WIDTH-120), 60)
                    angle = math.pi/4
                    cut = False
                    fall_velocity = 0.0
                    ball_x, ball_y = pendulum_pos(rope_anchor, rope_length, angle)

                    # --- FIXED basket spawn ---
                    basket_x = max(60, min(WIDTH - basket_w - 60, rope_anchor[0] - basket_w // 2 + random.randint(-40, 40)))
                    obstacles = generate_obstacles(max(1, level//2))

        if ball_y - ball_radius > HEIGHT:
            lives -= 1
            rope_anchor = (random.randint(120, WIDTH-120), 60)
            angle = math.pi/4
            cut = False
            fall_velocity = 0.0
            ball_x, ball_y = pendulum_pos(rope_anchor, rope_length, angle)

            # --- FIXED basket spawn ---
            basket_x = max(60, min(WIDTH - basket_w - 60, rope_anchor[0] - basket_w // 2 + random.randint(-40, 40)))
            obstacles = generate_obstacles(max(1, level//2))
            cv2.waitKey(150)
            if lives <= 0:
                return "gameover", score

        cv2.imshow("Cut the Rope", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return "quit", score


# --- Main Flow ---
cv2.namedWindow("Cut the Rope", cv2.WINDOW_NORMAL)  # persistent window
try:
    while True:
        start = start_menu()
        if not start:
            break
        result, final_score = game_loop()
        if result == "quit":
            break
        if result == "gameover":
            action = game_over_menu(final_score)
            if action == "quit":
                break
            elif action == "restart":
                continue
finally:
    cap.release()
    cv2.destroyAllWindows()
