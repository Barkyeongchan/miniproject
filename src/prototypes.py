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

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 웹캠 창 출력
    cv2.imshow("Video", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()