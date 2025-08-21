import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ========= 설정 =========
VIDEO_PATH = 0  # 웹캠 사용
SAVE_OUTPUT = False
OUTPUT_PATH = "optimized_output_perspective.mp4"
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

# ---------- 중앙 50% 크롭 ----------
def crop_center(frame, scale=0.5):
    h, w = frame.shape[:2]
    new_w, new_h = int(w*scale), int(h*scale)
    start_x, start_y = w//2 - new_w//2, h//2 - new_h//2
    return frame[start_y:start_y+new_h, start_x:start_x+new_w], (start_x, start_y)

# ---------- 깜빡이 여부 판단 (밝기 변화 기반) ----------
def check_turn_signal_any_color(obj_crop):
    gray = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity > 50  # 밝으면 깜빡이 있다고 판단

# ---------- ROI 제외 영역 탐지 ----------
def detect_side_objects(cropped, roi_mask):
    results = model(cropped, imgsz=320, verbose=False)
    side_objects = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                if np.all(roi_mask[y1:y2, x1:x2] == 0):
                    side_objects.append((x1, y1, x2, y2))
    return side_objects

# ---------- 원근감 기준 좌우 가장 가까운 객체 선택 ----------
def select_closest_objects_perspective(side_objects, frame_width):
    left_obj = None
    right_obj = None
    left_max_y = -1
    right_max_y = -1
    center_x = frame_width // 2

    for x1, y1, x2, y2 in side_objects:
        obj_cx = (x1 + x2) // 2
        obj_bottom = y2  # 원근감 기준: y2가 클수록 가까움

        if obj_cx < center_x:  # 좌측
            if obj_bottom > left_max_y:
                left_max_y = obj_bottom
                left_obj = (x1, y1, x2, y2)
        else:  # 우측
            if obj_bottom > right_max_y:
                right_max_y = obj_bottom
                right_obj = (x1, y1, x2, y2)

    selected = []
    if left_obj: selected.append(left_obj)
    if right_obj: selected.append(right_obj)
    return selected

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

        cropped, (ox, oy) = crop_center(frame, 0.5)

        # ROI 마스크 생성
        roi_mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        cv2.fillPoly(roi_mask, [np.array(trapezoid_pts[0]) - [ox, oy]], 1)

        # YOLO 탐지
        if frame_count % DETECTION_INTERVAL == 0:
            side_objects = detect_side_objects(cropped, roi_mask)
            prev_side_objects = select_closest_objects_perspective(side_objects, cropped.shape[1])

        # 박스 그리기 + 깜빡이 확인
        processed_crop = cropped.copy()
        for x1, y1, x2, y2 in prev_side_objects:
            obj_crop = processed_crop[y1:y2, x1:x2]
            blink = check_turn_signal_any_color(obj_crop)
            color = (0,255,0) if blink else (0,0,255)
            cv2.rectangle(processed_crop, (x1,y1), (x2,y2), color, 2)
            text = f"B:{int(blink)}"
            cv2.putText(processed_crop, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        frame[oy:oy+processed_crop.shape[0], ox:ox+processed_crop.shape[1]] = processed_crop
        processed_frame = visualize_only(frame, trapezoid_pts)

        frame_count += 1
        elapsed = time.time() - start_time
        fps_display = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(processed_frame, f"FPS: {fps_display:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        vis_resized = cv2.resize(processed_frame, (int(processed_frame.shape[1]*SCALE),
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
