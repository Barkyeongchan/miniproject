import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from collections import deque
import os
from threading import Thread

# ========= 설정 =========
VIDEO_PATH = 0  # 웹캠 사용
SCALE = 1.0
TRAPEZOID_TOP_Y = 0.8  # 사다리꼴 상단 Y축 좌표 (비율)
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 170
TRAPEZOID_BOTTOM_WIDTH = 400

EVENT_PRE_SEC = 3
EVENT_POST_SEC = 3
EVENT_IGNORE_SEC = 5
EVENT_SAVE_PATH = "../../img"
os.makedirs(EVENT_SAVE_PATH, exist_ok=True)

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# YOLO 모델 로드
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"YOLO 모델 로드 오류: {e}")
    model = None

torch.set_num_threads(4)

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

# ---------- 이벤트 판단 (OpenCV만 사용) ----------
def is_event_by_bbox(bbox, trapezoid_pts, top_hit_flags):
    x1, y1, x2, y2 = bbox

    # ROI 마스크 크기: ROI와 바운딩 박스를 모두 포함
    h = max(trapezoid_pts[:,:,1].max(), y2) + 5
    w = max(trapezoid_pts[:,:,0].max(), x2) + 5
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, trapezoid_pts, 1)

    event = False

    # 좌변 체크
    for yy in range(y1, y2+1):
        if yy >= h or x1 >= w:
            continue
        if roi_mask[yy, x1]:
            top_hit_flags['left'] = True
            event = True
            break

    # 우변 체크
    for yy in range(y1, y2+1):
        if yy >= h or x2 >= w:
            continue
        if roi_mask[yy, x2]:
            top_hit_flags['right'] = True
            event = True
            break

    # 상단: 좌/우 먼저 안 겹쳤으면 제외
    if not (top_hit_flags['left'] or top_hit_flags['right']):
        for xx in range(x1, x2+1):
            if xx >= w or y1 >= h:
                continue
            if roi_mask[y1, xx]:
                return False

    # 하단 체크
    if not event:
        for xx in range(x1, x2+1):
            if xx >= w or y2 >= h:
                continue
            if roi_mask[y2, xx]:
                event = True
                break

    return event

# ---------- 이벤트 영상 저장 ----------
def save_event_video(frames, fps):
    filename = os.path.join(EVENT_SAVE_PATH, f"event_{int(time.time())}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[INFO] 이벤트 영상 저장 완료: {filename}")

# ---------- 메인 루프 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오 열 수 없음")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_buffer = deque(maxlen=int(EVENT_PRE_SEC*fps))
    event_recording = False
    event_frames = []
    post_counter = 0
    last_event_time = 0

    ret, frame = cap.read()
    if not ret:
        print("첫 프레임 읽기 실패")
        return

    trapezoid_pts = calculate_fixed_trapezoid(frame.shape)
    frame_count = 0
    start_time = time.time()
    DETECTION_INTERVAL = 2
    prev_side_objects = []
    top_hit_flags = {'left': False, 'right': False}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_buffer.append(frame.copy())
        event = False

        # 차량 탐지
        if frame_count % DETECTION_INTERVAL == 0:
            side_objects = detect_side_objects(frame)
            prev_side_objects = select_closest_objects_perspective(side_objects, frame.shape[1])

        processed_frame = frame.copy()

        for idx, (side, (x1, y1, x2, y2)) in enumerate(prev_side_objects):
            # 이벤트 판단
            if is_event_by_bbox((x1, y1, x2, y2), trapezoid_pts, top_hit_flags):
                current_time = time.time()
                if current_time - last_event_time >= EVENT_IGNORE_SEC:
                    event = True
                    last_event_time = current_time
                    cv2.putText(processed_frame, "EVENT",
                                (frame.shape[1]//2 - 60, frame.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

            color = (0, 0, 255) if event else (255, 0, 0)
            cv2.rectangle(processed_frame, (x1,y1), (x2,y2), color, 2)

        # 이벤트 처리
        if event:
            if not event_recording:
                event_recording = True
                event_frames = list(frame_buffer)
                post_counter = 0

        if event_recording:
            event_frames.append(frame.copy())
            post_counter += 1
            if post_counter >= int(EVENT_POST_SEC*fps):
                Thread(target=save_event_video, args=(event_frames.copy(), fps)).start()
                event_recording = False
                event_frames = []

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

        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
