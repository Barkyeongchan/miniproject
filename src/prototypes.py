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