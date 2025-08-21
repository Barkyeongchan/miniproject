import cv2
import numpy as np
import time
from ultralytics import YOLO

# ========= 설정 =========
VIDEO_PATH = 0  # 웹캠 사용
SAVE_OUTPUT = False
OUTPUT_PATH = "fixed_30fps_output.mp4"
SCALE = 1.2

# 고정 사다리꼴 기본 설정값
TRAPEZOID_TOP_Y = 0.6
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 30
TRAPEZOID_BOTTOM_WIDTH = 400

# YOLO 모델 로드
try:
    model = YOLO('yolo11n', device="cuda:0")
except Exception as e:
    print(f"YOLO 모델 로드 중 오류: {e}")
    model = None

# ---------- 고정된 사다리꼴 좌표 ----------
def calculate_fixed_trapezoid(frame_shape):
    h, w = frame_shape[:2]
    center_x = w // 2
    top_y = int(h * TRAPEZOID_TOP_Y)
    bottom_y = int(h * TRAPEZOID_BOTTOM_Y)
    top_left_x = center_x - TRAPEZOID_TOP_WIDTH // 2
    top_right_x = center_x + TRAPEZOID_TOP_WIDTH // 2
    bottom_left_x = center_x - TRAPEZOID_BOTTOM_WIDTH // 2
    bottom_right_x = center_x + TRAPEZOID_BOTTOM_WIDTH // 2
    trapezoid_pts = np.array([[
        (bottom_left_x, bottom_y),
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bottom_right_x, bottom_y)
    ]], dtype=np.int32)
    return trapezoid_pts

# ---------- ROI 시각화 ----------
def visualize_only(frame, trapezoid_pts):
    overlay = np.zeros_like(frame, np.uint8)
    cv2.fillPoly(overlay, trapezoid_pts, (0, 255, 0))
    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

# ---------- 중앙 50% 영역 크롭 ----------
def crop_center(frame, scale=0.5):
    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    start_x = w // 2 - new_w // 2
    start_y = h // 2 - new_h // 2
    end_x = start_x + new_w
    end_y = start_y + new_h
    return frame[start_y:end_y, start_x:end_x], (start_x, start_y)

# ---------- 깜빡이 상태 판단 ----------
def check_turn_signal(car_crop):
    h, w = car_crop.shape[:2]
    left_blink = car_crop[:, :w//5]
    right_blink = car_crop[:, -w//5:]
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
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    ret, frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    trapezoid_points = calculate_fixed_trapezoid(frame.shape)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 중앙 50% 크롭
        cropped, (offset_x, offset_y) = crop_center(frame, scale=0.5)

        # YOLO 탐지 (중앙 영역)
        if model:
            results = model(cropped, verbose=False)
            processed_crop = cropped.copy()
            for r in results:
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    if int(cls) == 2:  # car class index (yolo11n 기준, 확인 필요)
                        x1, y1, x2, y2 = map(int, box)
                        car_crop = processed_crop[y1:y2, x1:x2]
                        left_on, right_on = check_turn_signal(car_crop)
                        # 상태 표시
                        color = (0,255,0) if left_on or right_on else (0,0,255)
                        cv2.rectangle(processed_crop, (x1,y1), (x2,y2), color, 2)
                        text = f"L:{int(left_on)} R:{int(right_on)}"
                        cv2.putText(processed_crop, text, (x1, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            processed_crop = cropped.copy()

        # 원본 프레임에 반영
        frame[offset_y:offset_y+processed_crop.shape[0],
              offset_x:offset_x+processed_crop.shape[1]] = processed_crop

        # ROI 시각화
        processed_frame = visualize_only(frame, trapezoid_points)

        # FPS 계산
        frame_count += 1
        elapsed_time = time.time() - start_time
        actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(processed_frame,
                    f"FPS: {actual_fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        vis_resized = cv2.resize(processed_frame,
                                 (int(processed_frame.shape[1]*SCALE),
                                  int(processed_frame.shape[0]*SCALE)))
        cv2.imshow("Drive", vis_resized)

        if writer is not None:
            writer.write(processed_frame)

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
