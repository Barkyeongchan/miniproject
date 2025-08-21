import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch

# ========= 설정 =========
VIDEO_PATH = 0  # 웹캠 사용
SAVE_OUTPUT = False
OUTPUT_PATH = "fixed_30fps_output.mp4"
SCALE = 1.0   # 렌더링 FPS 최적화 위해 1.0 유지
TRAPEZOID_TOP_Y = 0.6
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 30
TRAPEZOID_BOTTOM_WIDTH = 400

# YOLO 모델 로드
try:
    model = YOLO('yolov8n.pt')  # 가벼운 n 모델
except Exception as e:
    print(f"YOLO 모델 로드 중 오류: {e}")
    model = None

# CPU 멀티스레드 활용
torch.set_num_threads(4)

# ---------- 고정된 사다리꼴 좌표 ----------
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

# ---------- 깜빡이 상태 판단 ----------
def check_turn_signal(car_crop):
    h, w = car_crop.shape[:2]
    if h==0 or w==0:
        return False, False
    left_blink = car_crop[:, :max(1,w//5)]
    right_blink = car_crop[:, -max(1,w//5):]
    hsv_left = cv2.cvtColor(left_blink, cv2.COLOR_BGR2HSV)
    hsv_right = cv2.cvtColor(right_blink, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([10, 100, 200])
    upper_orange = np.array([25, 255, 255])
    left_mask = cv2.inRange(hsv_left, lower_orange, upper_orange)
    right_mask = cv2.inRange(hsv_right, lower_orange, upper_orange)
    left_on = cv2.countNonZero(left_mask) > 5
    right_on = cv2.countNonZero(right_mask) > 5
    return left_on, right_on

# ---------- 메인 루프 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        return

    writer = None
    if SAVE_OUTPUT:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    ret, frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    trapezoid_pts = calculate_fixed_trapezoid(frame.shape)

    frame_count = 0
    start_time = time.time()

    DETECTION_INTERVAL = 2  # 2프레임마다 YOLO 호출
    prev_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped, (ox, oy) = crop_center(frame, scale=0.5)

        # YOLO 탐지
        if model and frame_count % DETECTION_INTERVAL == 0:
            results = model(cropped, imgsz=320, verbose=False)
            prev_detections = []
            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls)==2:  # car
                        prev_detections.append(list(map(int, box)))

        # 직접 박스 그리기 + 깜빡이 확인
        processed_crop = cropped.copy()
        for box in prev_detections:
            x1,y1,x2,y2 = box
            car_crop = processed_crop[y1:y2, x1:x2]
            left_on, right_on = check_turn_signal(car_crop)
            color = (0,255,0) if left_on or right_on else (0,0,255)
            cv2.rectangle(processed_crop, (x1,y1), (x2,y2), color, 2)
            text = f"L:{int(left_on)} R:{int(right_on)}"
            cv2.putText(processed_crop, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 원본 반영
        frame[oy:oy+processed_crop.shape[0], ox:ox+processed_crop.shape[1]] = processed_crop

        # ROI
        processed_frame = visualize_only(frame, trapezoid_pts)

        # FPS
        frame_count += 1
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed if elapsed>0 else 0
        cv2.putText(processed_frame, f"FPS: {actual_fps:.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 출력
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
