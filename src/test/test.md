<details>
<summary>250814</summary>
<div markdown="1">

## 1. 웹캠 여는 기본 코드 생성

```python3
import cv2

# 카메라 캡쳐 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # ret = 프레임 읽기를 성공 여부를 나타냄
    if not ret:             # 프레임 읽기를 하지 못하면
        break               # 끝냄

    cv2.imshow("Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 2. 움직임을 감지하는 코드 추가

```python3
import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # 정적인 배경 학습 객체

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)  # 정적인 프레임과 비교하여 움직임을 감지
    
    # 노이즈 제거
    fgmask = cv2.medianBlur(fgmask, 5)

    # 움직임 영역 표시
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:  # 작은 잡음 무시
            
            print('움직임 발생')
            i = 1
            for i in range(100): # 반복문을 사용하여 이름 중복을 막음
                cap_img = os.path.join('../img', f'motion_detected{i}.jpg')         
                cv2.imwrite(cap_img, frame)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 웹캠 창 출력
    cv2.imshow("Video", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 3. 1초에 한 번 캡쳐하게 수정 - 실패함

```python3
import cv2
import os
import time

# 카메라 캡쳐 열기
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # 정적인 배경 학습 객체

while True:
    ret, frame = cap.read() # ret = 프레임 읽기를 성공 여부를 나타냄
    if not ret:             # 프레임 읽기를 하지 못하면
        break               # 끝냄

    fgmask = fgbg.apply(frame)  # 정적인 프레임과 비교하여 움직임을 감지
    
    # 노이즈 제거
    fgmask = cv2.medianBlur(fgmask, 5)

    # 움직임 영역 표시
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:  # 작은 잡음 무시

            print('움직임 발생')
            i = 1
            for i in range(100):
                cap_img = os.path.join('../img', f'motion_detected{i}.jpg')         
                cv2.imwrite(cap_img, frame)

            time.sleep(1.0)
            '''
            motion_detected1.jpg부터 motion_detected100.jpg까지 캡쳐 후 1초 멈춤 - 계획과 다름
            '''

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 웹캠 창 출력
    cv2.imshow("Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 4. 전역 카운터 변수를 사용해 문제 해결

```python3
import cv2
import os
import time

# 카메라 캡쳐 열기
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # 정적인 배경 학습 객체

# 이미지 저장시 이름 숫자 카운터 초기화
img_counter = 0

while True:
    ret, frame = cap.read() # ret = 프레임 읽기를 성공 여부를 나타냄
    if not ret:             # 프레임 읽기를 하지 못하면
        break               # 끝냄

    fgmask = fgbg.apply(frame)  # 정적인 프레임과 비교하여 움직임을 감지
    
    # 노이즈 제거
    fgmask = cv2.medianBlur(fgmask, 5)

    # 움직임 영역 표시
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False # 움직임이 감지되지 않으로 초기화

    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:  # 작은 잡음 무시
            motion_detected = True  # 움직임 감지
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion_detected:     
        cap_img = os.path.join('../img', f'motion_detected{img_counter}.jpg')         
        cv2.imwrite(cap_img, frame)
        print('움직임 발생')
        img_counter += 1

        time.sleep(1.0) # 1초 대기

    # 웹캠 창 출력
    cv2.imshow("Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 5. 디렉토리 자동 생성 코드 추가

```python3
import cv2
import os
import time

# 저장 경로 지정
save_dir = '../img'
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

# 카메라 캡쳐 열기
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # 정적인 배경 학습 객체

# 이미지 저장시 이름 숫자 카운터 초기화
img_counter = 0

while True:
    ret, frame = cap.read() # ret = 프레임 읽기를 성공 여부를 나타냄
    if not ret:             # 프레임 읽기를 하지 못하면
        break               # 끝냄

    fgmask = fgbg.apply(frame)  # 정적인 프레임과 비교하여 움직임을 감지
    
    # 노이즈 제거
    fgmask = cv2.medianBlur(fgmask, 5)

    # 움직임 영역 표시
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False # 움직임이 감지되지 않으로 초기화

    for cnt in contours:
        if cv2.contourArea(cnt) > 3000:  # 작은 잡음 무시
            motion_detected = True  # 움직임 감지
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if motion_detected:     
        cap_img = os.path.join(save_dir, f'motion_detected{img_counter}.jpg')         
        cv2.imwrite(cap_img, frame)
        print('움직임 발생')
        img_counter += 1

        time.sleep(1.0) # 1초 대기

    # 웹캠 창 출력
    cv2.imshow("Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

</div>
</details>