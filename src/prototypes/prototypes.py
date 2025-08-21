import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ========= 설정 =========
VIDEO_PATH = 0  # 웹캠 사용
SAVE_OUTPUT = False
OUTPUT_PATH = "blink_flashing_event_final.mp4"
SCALE = 1.0
TRAPEZOID_TOP_Y = 0.6
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 30
TRAPEZOID_BOTTOM_WIDTH = 400

# 차량 클래스
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# YOLO 모델 로드
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"YOLO 모델 로드 오류: {e}")
    model = None

torch.set_num_threads(4)  # CPU 스레드 설정

# ---------- LED 상태 저장 ----------
led_history = {}  # {obj_id_side: [0,1,0,1,...]}

# ---------- 고정 사다리꼴 ----------
def calculate_fixed_trapezoid(frame_shape):
    h, w = frame_shape[:2]
    cx = w // 2
    top_y = int(h * TRAPEZOID_TOP_Y)
    bottom_y = int(h * TRAPEZOID_BOTTOM_Y)
    top_left_x = cx - TRAPEZOID_TOP_WIDTH // 2
    top_right_x = cx + TRAPEZOID_TOP_WIDTH // 2
    bottom_left_x = cx - TRAPEZOID_BOTTOM_WIDTH // 2
    bottom_right_x = cx + TRAPEZOID_BOTTOM_WIDTH // 2
    return np.array([[ 
        (bottom_left_x, bottom_y),
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bottom_right_x, bottom_y)
    ]], dtype=np.int32)

# ---------- ROI 시각화 ----------
def visualize_only(frame, trapezoid_pts):
    overlay = np.zeros_like(frame, np.uint8)
    cv2.fillPoly(overlay, trapezoid_pts, (0, 255, 0))
    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

# ---------- LED 후보 영역 추출 ----------
def get_led_roi(obj_crop, side='left'):
    h, w = obj_crop.shape[:2]
    if side == 'left':
        roi = obj_crop[:, :w//5]
    else:
        roi = obj_crop[:, -w//5:]
    return roi

# ---------- LED 깜빡임 판정 ----------
def is_blinking(obj_id_side, led_roi):
    hsv = cv2.cvtColor(led_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10,100,200]), np.array([25,255,255]))
    bright = int(cv2.countNonZero(mask) > 5)
    
    history = led_history.get(obj_id_side, [])
    history.append(bright)
    if len(history) > 5:
        history = history[-5:]
    
    led_history[obj_id_side] = history
    
    # 깜빡임 판정: 최소 0→1→0 패턴 발생 여부
    if 1 in history and 0 in history and history.count(1) >= 1 and history.count(0) >= 1:
        return True
    return False

# ---------- 차량 탐지 ----------
def detect_side_objects(frame):
    results = model(frame, imgsz=320, verbose=False)
    side_objects = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                side_objects.append((x1, y1, x2, y2))
    return side_objects

# ---------- 좌우 가장 가까운 차량 ----------
def select_closest_objects_perspective(side_objects, frame_width):
    left_obj, right_obj = None, None
    left_max_y, right_max_y = -1, -1
    center_x = frame_width // 2
    for x1, y1, x2, y2 in side_objects:
        obj_cx = (x1 + x2) // 2
        obj_bottom = y2
        if obj_cx < center_x:
            if obj_bottom > left_max_y:
                left_max_y = obj_bottom
                left_obj = (x1, y1, x2, y2)
        else:
            if obj_bottom > right_max_y:
                right_max_y = obj_bottom
                right_obj = (x1, y1, x2, y2)
    selected = []
    if left_obj: selected.append(('left', left_obj))
    if right_obj: selected.append(('right', right_obj))
    return selected

# ---------- 사다리꼴 ROI 안에 있는지 확인 ----------
def is_in_trapezoid(obj_box, trapezoid_pts):
    x1, y1, x2, y2 = obj_box
    obj_cx, obj_cy = (x1 + x2)//2, (y1 + y2)//2
    return cv2.pointPolygonTest(trapezoid_pts[0], (obj_cx, obj_cy), False) >= 0

# ---------- 메인 루프 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오 열 수 없음")
        return

    writer = None
    if SAVE_OUTPUT:
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    ret, frame = cap.read()
    if not ret:
        print("첫 프레임 읽기 실패")
        return

    trapezoid_pts = calculate_fixed_trapezoid(frame.shape)
    frame_count = 0
    start_time = time.time()
    DETECTION_INTERVAL = 2
    prev_side_objects = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % DETECTION_INTERVAL == 0:
            side_objects = detect_side_objects(frame)
            prev_side_objects = select_closest_objects_perspective(side_objects, frame.shape[1])

        processed_frame = frame.copy()
        for idx, (side, (x1, y1, x2, y2)) in enumerate(prev_side_objects):
            obj_crop = processed_frame[y1:y2, x1:x2]
            roi = get_led_roi(obj_crop, side)
            obj_id_side = f"{idx}_{side}"
            blink = is_blinking(obj_id_side, roi)
            color = (0,255,0) if blink else (0,0,255)
            cv2.rectangle(processed_frame, (x1,y1), (x2,y2), color, 2)
            text = "BLINK" if blink else "OFF"
            cv2.putText(processed_frame, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 사다리꼴 ROI 이벤트 표시
            if not blink and is_in_trapezoid((x1, y1, x2, y2), trapezoid_pts):
                cv2.putText(processed_frame, "EVENT",
                            (frame.shape[1]//2 - 60, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        processed_frame = visualize_only(processed_frame, trapezoid_pts)

        frame_count += 1
        elapsed = time.time() - start_time
        fps_display = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(processed_frame, f"FPS: {fps_display:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        vis_resized = cv2.resize(processed_frame,
                                 (int(processed_frame.shape[1]*SCALE),
                                  int(processed_frame.shape[0]*SCALE)))
        cv2.imshow("Drive", vis_resized)
        if writer: writer.write(processed_frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
