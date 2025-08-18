import cv2
import os
import time

# 저장 경로 지정
save_dir = '../../img'
os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

# 카메라 캡쳐 열기
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # 정적인 배경 학습 객체

# 이미지 저장시 이름 숫자 카운터 초기화
img_counter = 0

last_capture_time = 0  # 마지막 캡처 시각
capture_interval = 1.0  # 캡처 간격(초)

# FPS 측정용 변수
prev_time = time.time()

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

    # 현재 시각
    current_time = time.time()

    # FPS 계산
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    # 프레임 위에 FPS 표시
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # 움직임이 감지되고, 마지막 저장 시각에서 일정 시간이 지났으면 저장
    if motion_detected and (current_time - last_capture_time) >= capture_interval:     
        cap_img = os.path.join(save_dir, f'motion_detected{img_counter}.jpg')         
        cv2.imwrite(cap_img, frame)
        print('움직임 발생')
        img_counter += 1
        last_capture_time = current_time    # 마지막 캡처 시각 갱신

    # 웹캠 창 출력
    cv2.imshow("Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()