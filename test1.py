import cv2
import numpy as np
import time
import os
from pynput import keyboard

ESC_PRESSED = False

def on_press(key):
    global ESC_PRESSED
    try:
        if key == keyboard.Key.esc:
            ESC_PRESSED = True
    except:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

# ------------------------------
# Try import mediapipe
# ------------------------------
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


# ------------------------------
# USER CONFIG
# ------------------------------
BG_IMG = "background.jpg"
RECORD_DIR = "recordings"
BENCHMARK_DURATION = 8
PROC_WIDTH = 640


# ------------------------------
# Helpers
# ------------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def resize_keep_aspect(frame, width=PROC_WIDTH):
    h, w = frame.shape[:2]
    if w == 0: return frame
    if w == width: return frame
    r = width / float(w)
    return cv2.resize(frame, (width, int(h*r)), interpolation=cv2.INTER_AREA)


# ------------------------------
# Segmentation wrapper
# ------------------------------
class SegWrapper:
    def __init__(self):
        self.mp_seg = None
        self.bg_sub = None

        if MP_AVAILABLE:
            try:
                self.mp_selfie = mp.solutions.selfie_segmentation
                self.mp_seg = self.mp_selfie.SelfieSegmentation(model_selection=1)
            except:
                self.mp_seg = None

        try:
            self.bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=25, detectShadows=False)
        except:
            self.bg_sub = None

    def get_mask(self, frame):
        h, w = frame.shape[:2]

        if self.mp_seg is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.mp_seg.process(rgb)
                mask = res.segmentation_mask.astype("float32")
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h))
                return mask
            except:
                pass

        if self.bg_sub is not None:
            fg = self.bg_sub.apply(frame)
            fg = cv2.GaussianBlur(fg, (5,5), 0)
            _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            return (th.astype("float32") / 255.0)

        return np.ones((h, w), dtype="float32")

    def close(self):
        if self.mp_seg is not None:
            try:
                self.mp_seg.close()
            except:
                pass


# ------------------------------
# Modes
# ------------------------------
def mode_original(frame, info):
    out = frame.copy()
    cv2.putText(out, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

def mode_ar(frame, mask, bg):
    h, w = frame.shape[:2]
    mask3 = np.dstack([mask>0.5]*3).astype("uint8")
    bg = cv2.resize(bg, (w, h)) if bg is not None else np.zeros_like(frame)
    fg = (frame * mask3)
    bgp = (bg * (1-mask3))
    return fg + bgp

def mode_position(frame, mask, info):
    h, w = frame.shape[:2]
    binmask = ((mask>0.5)*255).astype("uint8")
    contours, _ = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pos = "No person"
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 500:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                if cx < w/3: pos = "Left"
                elif cx < 2*w/3: pos = "Center"
                else: pos = "Right"
            cv2.drawContours(frame, [c], -1, (0,255,0), 2)

    cv2.putText(frame, f"Position: {pos}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255),2)
    cv2.putText(frame, info, (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
    return frame

def mode_blur(frame, mask, info):
    blurred = cv2.GaussianBlur(frame, (35,35), 0)
    mask3 = np.dstack([mask>0.5]*3).astype("uint8")
    out = blurred.copy()
    out = (blurred*(1-mask3) + frame*mask3).astype("uint8")
    cv2.putText(out, info, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
    return out

def mode_heatmap(frame, mask, info):
    m = np.clip(mask*255, 0, 255).astype("uint8")
    heat = cv2.applyColorMap(m, cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 0.5, heat, 0.5, 0)
    cv2.putText(out, info, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
    return out


def run_benchmark(cap, seg, duration):
    global ESC_PRESSED
    ESC_PRESSED = False

    print(f"[BENCHMARK] Running {duration} seconds... Press ESC to stop.")

    start = time.time()
    frames = 0
    timestamps = []

    win = "Benchmark"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        if ESC_PRESSED:
            print("[BENCHMARK] ESC detected, stopping.")
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_keep_aspect(frame)

        _ = seg.get_mask(frame)  # workload
        frames += 1
        timestamps.append(time.time())

        cv2.imshow(win, frame)
        cv2.waitKey(1)

        if time.time() - start >= duration:
            break

    cv2.destroyWindow(win)

    if len(timestamps) < 2:
        print("[BENCHMARK] Not enough frames.")
        return

    intervals = np.diff(np.array(timestamps))
    intervals = intervals[intervals > 0]
    fps = 1 / intervals

    print(f"[BENCHMARK] Frames: {frames}")
    print(f"[BENCHMARK] Mean FPS: {fps.mean():.2f}")
    print(f"[BENCHMARK] Median FPS: {np.median(fps):.2f}")
    print(f"[BENCHMARK] Min FPS: {fps.min():.2f}")
    print(f"[BENCHMARK] Max FPS: {fps.max():.2f}")
    print("[BENCHMARK] Done.")


# ------------------------------
# MAIN
# ------------------------------
def main():
    global ESC_PRESSED

    ensure_dir(RECORD_DIR)
    seg = SegWrapper()

    bg = cv2.imread(BG_IMG) if os.path.exists(BG_IMG) else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: webcam not found.")
        return

    mode = 1
    paused = False
    recording = False
    writer = None
    prev_time = time.time()
    smooth_fps = 0

    print("Ready. ESC = exit ANYTIME.")

    while True:
        if ESC_PRESSED:
            print("[EXIT] ESC detected — quitting.")
            break

        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame = resize_keep_aspect(frame)

        mask = seg.get_mask(frame) if mode in (1,2,3,4) else np.zeros(frame.shape[:2])
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        info = f"Mode {mode} | ESC=exit"

        if mode == 0: out = mode_original(frame, info)
        elif mode == 1: out = mode_ar(frame, mask, bg)
        elif mode == 2: out = mode_position(frame.copy(), mask, info)
        elif mode == 3: out = mode_blur(frame, mask, info)
        elif mode == 4: out = mode_heatmap(frame, mask, info)
        elif mode == 5:
            run_benchmark(cap, seg, BENCHMARK_DURATION)
            continue

        ct = time.time()
        fps = 1 / (ct - prev_time) if ct - prev_time > 0 else 0
        smooth_fps = smooth_fps * 0.8 + fps * 0.2
        prev_time = ct
        cv2.putText(out, f"FPS:{int(smooth_fps)}", (10,out.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

        cv2.imshow("Intelligent Vision Demo", out)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5')):
            mode = int(chr(key))

        if key == ord('p'):
            paused = not paused

        if key == ord('b'):
            if os.path.exists(BG_IMG):
                bg = cv2.imread(BG_IMG)
                print("[BG] Reloaded")

    cap.release()
    cv2.destroyAllWindows()
    seg.close()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
