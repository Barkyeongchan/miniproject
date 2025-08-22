# 미니 프로젝트 : 교통법규 위반 신고 간편화 프로그램

## 목차

1. 개발 환경
   - 하드웨어
   - 개발 환경 및 사용 기술

2. 주제 선정
   - 프로젝트 주제
   - 관련 제품 링크
   - 기존 제품의 한계점과 개선점

3. 진행 상황

4. 결과

## 1. 개발 환경

<details>
<summary></summary>
<div markdown="1">

## **1-1. 하드웨어**

개인 노트북
- GPU : NVIDIA GeForce 940MX

웹캠
- Logitech C920

## **1-2. 개발 환경 및 사용 기술**

| 언어 | 라이브러리 / 프레임워크 | 개발 도구 | 운영체제 |
|------|------------------------|-----------|-----------|
| ![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white) | ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen?logo=opencv&logoColor=white) ![YOLO](https://img.shields.io/badge/YOLO-v11-orange) | ![VSCode](https://img.shields.io/badge/VSCode-blueviolet?logo=visual-studio-code&logoColor=white) | ![Windows](https://img.shields.io/badge/Windows-10-lightgrey?logo=windows&logoColor=white) |

</div>
</details>

## 2. 주제 선정

<details>
<summary></summary>
<div markdown="1">

## **2-1. 프로젝트 주제**

**주행 중 교통법규 위반 이벤트 발생시 자동으로 영상 캡쳐 후 지정된 위치로 영상 발송**

## **2-2. 관련 제품 링크**

_국내 블랙박스 브랜드 점유율 TOP 2 기업_

[아이나비](https://www.inavi.com/)

[파인뷰](http://www.fine-drive.com/defaults/index.do)

## **2-3. 기존 제품의 한계점과 개선점**

**[1. 한계점]**

기존의 블랙박스는 이벤트 발생(교통법규 위반)시 **임의로 제품이나 차량에 충격을 가하여 이벤트 순간을 특정**하거나,

주행이 끝난 뒤 스스로 이벤트 발생 순간을 확인해야함.

**[2. 개선점]**

이벤트 발생 순간을 **openCV와 YOLO를 통해 자동으로 인식**하고, 해당 부분의 영상을 캡쳐하여 지정한 위치(메일, 공유 폴더 등)로 자동으로 전송함.

</div>
</details>

## 3. 진행 상황

<details>
<summary></summary>
<div markdown="1">

### **[250814]**

[최소 구현 목표 변환](https://docs.google.com/document/d/1KR1Ek3QEK2PorDblV3TpiJ__KWXGa5N0oOg1SPpD4f0/edit?tab=t.2lpizc91xvge)에 맞춰 코드 생성

[test.md](src/test/test.md)

### **[250818]**

[테스트 주도 개발(TDD)](https://docs.google.com/document/d/1PHuT9a7qTqCdTEbbHtY2iAA3BITojjOxiW3L5UlMCSY/edit?tab=t.0#heading=h.qun0cyfrhp5e)방법 적용

### **[250819]**

차선 검출 방법 수정 후 확인 반복

### **[250820]**

차선을 검출하는 방법에서 차량 앞 일정 구역을 관심영역으로 지정하는 방법으로 바꿈

### **[250821]**

이벤트 발생 목표를 방향지시등 미점멸 차선변경으로 변경 후 이벤트 발생 검출 완료

코드 작동 확인

</div>
</details>

## 4. 결과

<details>
<summary></summary>
<div markdown="1">

### **[테스트 환경]**

<img width="654" height="368" alt="image" src="https://raw.githubusercontent.com/Barkyeongchan/miniproject/refs/heads/main/assets/final.gif" />

### **[예제 영상과 실제 적용 화면]**

<img width="640" height="480" alt="image" src="https://raw.githubusercontent.com/Barkyeongchan/miniproject/refs/heads/main/assets/event_1755761493.gif" />

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/6a302dc8-262d-49a9-b43b-a31d0d928c61" />

### **[전체 코드]**

```python
import cv2
import numpy as np
import time
from ultralytics import YOLO
import torch
from collections import deque
import os
from threading import Thread

# ======================= 1. 전역 설정 =======================
# 이 섹션에서는 프로그램 전반에 사용되는 변수들을 정의합니다.
# 경로, 크기, 시간 등을 쉽게 조정할 수 있습니다.
VIDEO_PATH = 0  # 0 = 웹캠 사용, "파일명.mp4" = 비디오 파일 경로
SCALE = 1.0  # 최종 화면의 확대/축소 비율 (1.0 = 원본 크기)

# 사다리꼴 ROI(관심 영역) 설정. 차량 진입을 감지할 도로 영역입니다.
# 비율(0.0 ~ 1.0)과 픽셀 너비를 조합하여 원하는 구역을 지정할 수 있습니다.
TRAPEZOID_TOP_Y = 0.8  # 사다리꼴 상단 Y축 좌표 (영상 높이의 80% 지점)
TRAPEZOID_BOTTOM_Y = 1.0  # 사다리꼴 하단 Y축 좌표 (영상 맨 아래)
TRAPEZOID_TOP_WIDTH = 170  # 상단 선의 너비 (픽셀)
TRAPEZOID_BOTTOM_WIDTH = 400  # 하단 선의 너비 (픽셀)

# 이벤트(사고 위험) 영상 저장 관련 설정
EVENT_PRE_SEC = 3  # 이벤트 발생 전 3초간의 영상을 저장
EVENT_POST_SEC = 3  # 이벤트 발생 후 3초간의 영상을 저장
EVENT_IGNORE_SEC = 5  # 이벤트 발생 후 5초 동안은 추가 이벤트 감지를 무시하여 중복 저장을 방지
EVENT_SAVE_PATH = "../../img"  # 이벤트 영상이 저장될 폴더 경로
os.makedirs(EVENT_SAVE_PATH, exist_ok=True)  # 폴더가 없으면 새로 생성

# YOLO 모델이 탐지할 객체 클래스 목록 (COCO 데이터셋 기준)
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# YOLO 모델 로드: 'yolov8n.pt'는 가장 작고 빠른 모델입니다.
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"YOLO 모델 로드 오류: {e}")
    model = None

# YOLO 연산 속도 개선 (CPU 연산 최적화)
# PyTorch가 모델 추론 시 사용할 스레드 수를 4개로 제한하여 불필요한 자원 낭비를 막고 효율을 높입니다.
torch.set_num_threads(4)

# ======================= 2. 유틸리티 함수들 =======================

def calculate_fixed_trapezoid(frame_shape):
    """
    영상 크기를 기준으로 사다리꼴 ROI의 픽셀 좌표를 계산합니다.
    frame_shape: (높이, 너비, 채널) 형태의 튜플
    반환값: OpenCV의 fillPoly 함수에 사용할 수 있는 numpy 배열 형태의 사다리꼴 좌표
    """
    h, w = frame_shape[:2]
    cx = w // 2  # 영상의 중심 X 좌표
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

def visualize_only(frame, trapezoid_pts):
    """
    원본 영상에 사다리꼴 ROI 영역을 반투명한 녹색으로 덧씌워 보여줍니다.
    frame: 원본 영상 프레임
    trapezoid_pts: 사다리꼴의 픽셀 좌표
    """
    overlay = np.zeros_like(frame, np.uint8)
    cv2.fillPoly(overlay, trapezoid_pts, (0, 255, 0))  # 사다리꼴 영역을 녹색으로 채움
    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)  # 원본과 오버레이를 섞어서 반투명 효과를 냄

def detect_side_objects(frame):
    """
    YOLOv8 모델을 사용해 영상 프레임 내의 모든 차량을 탐지합니다.
    - YOLO 모델의 입력 이미지 크기를 320x320으로 줄여 연산 속도를 높였습니다.
    - 'verbose=False' 옵션으로 불필요한 콘솔 출력을 끕니다.
    frame: 현재 영상 프레임
    반환값: 탐지된 차량들의 바운딩박스 리스트 [(x1, y1, x2, y2), ...]
    """
    results = model(frame, imgsz=320, verbose=False)
    side_objects = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box)
                side_objects.append((x1, y1, x2, y2))
    return side_objects

def select_closest_objects_perspective(side_objects, frame_width):
    """
    탐지된 차량들 중 좌/우측에서 가장 가까이 있는 차량(바운딩박스 Y좌표가 가장 큰)을 선택합니다.
    원근법에 따라, 화면 아래쪽에 있을수록 더 가까운 차량으로 판단합니다.
    side_objects: 탐지된 모든 차량의 바운딩박스 리스트
    frame_width: 영상의 너비
    반환값: ('left' 또는 'right', 바운딩박스) 튜플의 리스트
    """
    left_obj, right_obj = None, None
    left_max_y, right_max_y = -1, -1
    center_x = frame_width // 2
    for x1, y1, x2, y2 in side_objects:
        obj_cx = (x1 + x2) // 2
        obj_bottom = y2
        if obj_cx < center_x:  # 화면 왼쪽 영역의 차량
            if obj_bottom > left_max_y:
                left_max_y = obj_bottom
                left_obj = (x1, y1, x2, y2)
        else:  # 화면 오른쪽 영역의 차량
            if obj_bottom > right_max_y:
                right_max_y = obj_bottom
                right_obj = (x1, y1, x2, y2)
    selected = []
    if left_obj: selected.append(('left', left_obj))
    if right_obj: selected.append(('right', right_obj))
    return selected

def is_event_by_bbox(bbox, trapezoid_pts, top_hit_flags):
    """
    주어진 바운딩박스(bbox)가 사다리꼴 ROI에 진입했는지 판단합니다.
    - 바운딩박스의 좌/우/하단 선이 ROI와 겹치는지를 체크합니다.
    - 상단 선이 겹치는 경우는 좌/우선이 먼저 겹치지 않았을 경우 무시합니다. 이는 멀리 있는 차량이 ROI 상단에 걸리는 것을 방지하기 위함입니다.
    bbox: 차량의 바운딩박스 (x1, y1, x2, y2)
    trapezoid_pts: 사다리꼴 ROI 좌표
    top_hit_flags: 상단 겹침 여부를 기록하는 딕셔너리
    반환값: 이벤트 발생 여부 (True/False)
    """
    x1, y1, x2, y2 = bbox
    # ROI와 바운딩박스를 포함할 만큼 충분히 큰 마스크 생성
    h = max(trapezoid_pts[:,:,1].max(), y2) + 5
    w = max(trapezoid_pts[:,:,0].max(), x2) + 5
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, trapezoid_pts, 1) # ROI 영역을 1로 채움
    event = False

    # 좌변/우변 겹침 체크
    # 차량이 옆에서 진입하는 경우를 감지
    for yy in range(y1, y2+1):
        if yy >= h or x1 >= w: continue
        if roi_mask[yy, x1]:
            top_hit_flags['left'] = True
            event = True
            break

    if not event:
        for yy in range(y1, y2+1):
            if yy >= h or x2 >= w: continue
            if roi_mask[yy, x2]:
                top_hit_flags['right'] = True
                event = True
                break

    # 상단 겹침 체크: 좌/우변이 먼저 겹치지 않았으면 상단 겹침은 무시
    if not (top_hit_flags['left'] or top_hit_flags['right']):
        for xx in range(x1, x2+1):
            if xx >= w or y1 >= h: continue
            if roi_mask[y1, xx]:
                return False

    # 하단 겹침 체크: 차량이 ROI에 완전히 진입했는지 감지
    if not event:
        for xx in range(x1, x2+1):
            if xx >= w or y2 >= h: continue
            if roi_mask[y2, xx]:
                event = True
                break
    return event

def save_event_video(frames, fps):
    """
    이벤트 발생 시, 미리 버퍼링된 영상 프레임들을 하나의 동영상 파일로 저장합니다.
    - 이 함수는 메인 루프를 방해하지 않도록 별도의 **스레드(Thread)**로 실행됩니다.
    frames: 저장할 영상 프레임들의 리스트
    fps: 프레임 속도
    """
    filename = os.path.join(EVENT_SAVE_PATH, f"event_{int(time.time())}.mp4")
    h, w = frames[0].shape[:2]
    # MP4V 코덱을 사용하여 동영상 파일로 저장
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"[INFO] 이벤트 영상 저장 완료: {filename}")

# ======================= 3. 메인 실행 루프 =======================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오 열 수 없음")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # FPS 정보 가져오기, 실패 시 기본값 30
    # 이벤트 발생 전 영상을 저장하기 위한 버퍼
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
    # YOLO 모델을 매 프레임마다 실행하지 않고 2프레임마다 한 번씩만 실행하여 FPS를 높입니다.
    DETECTION_INTERVAL = 2
    prev_side_objects = []
    top_hit_flags = {'left': False, 'right': False}

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_buffer.append(frame.copy())
        event = False

        # 차량 탐지 (주기적으로 실행)
        if frame_count % DETECTION_INTERVAL == 0:
            side_objects = detect_side_objects(frame)
            prev_side_objects = select_closest_objects_perspective(side_objects, frame.shape[1])

        processed_frame = frame.copy()

        # 바운딩박스 체크 및 이벤트 판단
        for side, bbox in prev_side_objects:
            if is_event_by_bbox(bbox, trapezoid_pts, top_hit_flags):
                current_time = time.time()
                # 이벤트 무시 시간(EVENT_IGNORE_SEC)이 지나야만 새로운 이벤트로 인식
                if current_time - last_event_time >= EVENT_IGNORE_SEC:
                    event = True
                    last_event_time = current_time

            # 바운딩박스 색상 표시: 이벤트 발생 시 빨간색, 아닐 경우 파란색
            color = (0, 0, 255) if event else (255, 0, 0)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

        # 이벤트 발생 시 화면 중앙에 "EVENT" 텍스트 표시
        if event:
            cv2.putText(processed_frame, "EVENT",
                        (processed_frame.shape[1] // 2 - 60, processed_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # 이벤트 영상 저장 처리
        if event and not event_recording:
            event_recording = True
            # 이벤트 발생 전 프레임들을 버퍼에서 가져와 저장 리스트에 추가
            event_frames = list(frame_buffer)
            post_counter = 0

        if event_recording:
            # 이벤트 발생 후 프레임들을 저장 리스트에 추가
            event_frames.append(frame.copy())
            post_counter += 1
            # EVENT_POST_SEC 시간만큼 프레임이 쌓였으면 동영상 저장
            if post_counter >= int(EVENT_POST_SEC * fps):
                # 영상 저장을 메인 스레드와 분리하여 처리
                Thread(target=save_event_video, args=(event_frames.copy(), fps)).start()
                event_recording = False
                event_frames = []

        # ROI 시각화
        processed_frame = visualize_only(processed_frame, trapezoid_pts)

        # FPS 계산 및 화면 표시
        frame_count += 1
        elapsed = time.time() - start_time
        fps_display = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(processed_frame, f"FPS: {fps_display:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 화면에 결과 표시
        vis_resized = cv2.resize(processed_frame,
                                 (int(processed_frame.shape[1] * SCALE),
                                  int(processed_frame.shape[0] * SCALE)))
        cv2.imshow("Drive", vis_resized)

        # 'q' 또는 ESC 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

</div>
</details>
