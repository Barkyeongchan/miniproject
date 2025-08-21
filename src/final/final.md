🚗 차량 이벤트 감지 시스템

이 프로젝트는 웹캠 또는 비디오 파일을 이용하여 도로에서 차량 이벤트를 실시간으로 감지하고, 이벤트 전후 영상을 자동으로 저장하는 시스템입니다.
YOLO 객체 탐지 모델과 OpenCV를 활용합니다.

1. 사용 라이브러리
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from collections import deque
import os
from threading import Thread


cv2: 영상 처리 라이브러리(OpenCV)

numpy: 배열 및 수치 계산

time: 시간 측정 및 지연

YOLO: 객체 탐지 모델

torch: PyTorch, YOLO 연산 최적화

deque: 이벤트 전후 프레임 저장용 큐

os: 폴더 생성 및 파일 처리

Thread: 이벤트 영상 저장을 비동기 처리

2. 주요 설정 값
VIDEO_PATH = 0  # 0 = 웹캠 사용, 문자열 = 비디오 파일 경로
SCALE = 1.0     # 화면 확대/축소 비율

# 사다리꼴 ROI 설정 (도로 영역)
TRAPEZOID_TOP_Y = 0.8
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 170
TRAPEZOID_BOTTOM_WIDTH = 400

# 이벤트 저장 관련
EVENT_PRE_SEC = 3       # 이벤트 전 영상 저장 시간
EVENT_POST_SEC = 3      # 이벤트 후 영상 저장 시간
EVENT_IGNORE_SEC = 5    # 이벤트 발생 후 무시 시간
EVENT_SAVE_PATH = "../../img"
os.makedirs(EVENT_SAVE_PATH, exist_ok=True)

# YOLO 차량 클래스 (COCO 기준)
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck


사다리꼴 ROI: 차량 감지 관심 영역

EVENT_PRE_SEC / POST_SEC: 이벤트 전후 영상 저장 시간

EVENT_IGNORE_SEC: 연속 이벤트 발생 방지 시간

VEHICLE_CLASSES: YOLO COCO 기준 차량 클래스

3. YOLO 모델 로드 및 최적화
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"YOLO 모델 로드 오류: {e}")
    model = None

torch.set_num_threads(4)


YOLO 모델을 불러와 객체 탐지 준비

CPU 스레드 제한으로 연산 속도 최적화

4. 주요 함수 설명
4-1. ROI 사다리꼴 좌표 계산
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


프레임 크기에 맞춰 사다리꼴 ROI 좌표 계산

차량 박스와 겹치는지 확인할 때 사용

4-2. ROI 시각화
def visualize_only(frame, trapezoid_pts):
    overlay = np.zeros_like(frame, np.uint8)
    cv2.fillPoly(overlay, trapezoid_pts, (0, 255, 0))
    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)


ROI 영역을 녹색 반투명으로 표시

cv2.addWeighted로 원본 영상과 합성

4-3. 차량 탐지
def detect_side_objects(frame):
    results = model(frame, imgsz=320, verbose=False)
    side_objects = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                side_objects.append((x1, y1, x2, y2))
    return side_objects


YOLO를 이용하여 차량 객체 탐지

화면 내 좌표 (x1, y1, x2, y2) 저장

4-4. 좌/우 도로에서 가장 가까운 차량 선택
def select_closest_objects_perspective(side_objects, frame_width):
    left_obj, right_obj = None, None
    left_max_y, right_max_y = -1, -1
    center_x = frame_width // 2
    for x1, y1, x2, y2 in side_objects:
        obj_cx = (x1 + x2) // 2
        obj_bottom = y2
        if obj_cx < center_x and obj_bottom > left_max_y:
            left_max_y = obj_bottom
            left_obj = (x1, y1, x2, y2)
        elif obj_cx >= center_x and obj_bottom > right_max_y:
            right_max_y = obj_bottom
            right_obj = (x1, y1, x2, y2)

    selected = []
    if left_obj: selected.append(('left', left_obj))
    if right_obj: selected.append(('right', right_obj))
    return selected


화면 중심 기준 좌/우 도로 구분

가장 아래쪽(y 최대) 차량 선택 → 도로에서 가장 가까운 차량

4-5. 이벤트 판단
def is_event_by_bbox(bbox, trapezoid_pts, top_hit_flags):
    x1, y1, x2, y2 = bbox
    h = max(trapezoid_pts[:,:,1].max(), y2) + 5
    w = max(trapezoid_pts[:,:,0].max(), x2) + 5
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, trapezoid_pts, 1)
    event = False

    # 좌변 체크
    for yy in range(y1, y2+1):
        if yy >= h or x1 >= w: continue
        if roi_mask[yy, x1]:
            top_hit_flags['left'] = True
            event = True
            break

    # 우변 체크
    for yy in range(y1, y2+1):
        if yy >= h or x2 >= w: continue
        if roi_mask[yy, x2]:
            top_hit_flags['right'] = True
            event = True
            break

    # 상단: 좌/우 먼저 안 겹쳤으면 제외
    if not (top_hit_flags['left'] or top_hit_flags['right']):
        for xx in range(x1, x2+1):
            if xx >= w or y1 >= h: continue
            if roi_mask[y1, xx]:
                return False

    # 하단 체크
    if not event:
        for xx in range(x1, x2+1):
            if xx >= w or y2 >= h: continue
            if roi_mask[y2, xx]:
                event = True
                break

    return event


바운딩 박스와 ROI 겹침 여부로 이벤트 판단

상단 겹침은 좌/우 먼저 안 겹쳤으면 무시

하단이 겹치면 이벤트 발생

4-6. 이벤트 영상 저장
def save_event_video(frames, fps):
    filename = os.path.join(EVENT_SAVE_PATH, f"event_{int(time.time())}.mp4")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[INFO] 이벤트 영상 저장 완료: {filename}")


이벤트 발생 시 프레임들을 영상으로 저장

스레드 사용 → 메인 루프 멈추지 않음

5. 메인 루프
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
        if not ret: break

        frame_buffer.append(frame.copy())
        event = False

        # 차량 탐지 (주기적)
        if frame_count % DETECTION_INTERVAL == 0:
            side_objects = detect_side_objects(frame)
            prev_side_objects = select_closest_objects_perspective(side_objects, frame.shape[1])

        processed_frame = frame.copy()

        # 바운딩박스 체크 & 이벤트 판단
        for side, bbox in prev_side_objects:
            if is_event_by_bbox(bbox, trapezoid_pts, top_hit_flags):
                current_time = time.time()
                if current_time - last_event_time >= EVENT_IGNORE_SEC:
                    event = True
                    last_event_time = current_time

            # 박스 색상 표시
            color = (0, 0, 255) if event else (255, 0, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(processed_frame, (x1,y1), (x2,y2), color, 2)

        # 이벤트 발생 시 화면 중앙 표시
        if event:
            cv2.putText(processed_frame, "EVENT",
                        (processed_frame.shape[1]//2 - 60, processed_frame.shape[0]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)

        # 이벤트 영상 저장 처리
        if event and not event_recording:
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

        # ROI 시각화
        processed_frame = visualize_only(processed_frame, trapezoid_pts)

        # FPS 표시
        frame_count += 1
        elapsed = time.time() - start_time
        fps_display = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(processed_frame, f"FPS: {fps_display:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # 화면 표시
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


프레임 읽기: 웹캠 또는 영상 파일

frame_buffer: 이벤트 전후 영상 저장용 큐

DETECTION_INTERVAL: 몇 프레임마다 차량 탐지할지 설정

바운딩박스 색상: 이벤트 발생 시 빨강, 미발생 시 파랑

FPS 표시: 화면 좌상단

이벤트 저장: 비동기 스레드 사용

6. 요약

YOLO를 활용한 차량 객체 탐지

사다리꼴 ROI로 도로 영역 지정

좌/우 도로에서 가장 가까운 차량 이벤트 감지

이벤트 전후 영상 자동 저장

FPS와 ROI 시각화로 실시간 모니터링 가능

멀티스레드로 영상 저장 시 프레임 누락 방지
