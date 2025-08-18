import cv2
import numpy as np

VIDEO_PATH = "../../assets/drive_sample.mp4"
SCALE = 0.6

# 가우시안, 모폴로지 커널
GAUSS_KSIZE = 5
MORPH_K = 5

def detect_road(frame):
    h, w = frame.shape[:2]
    
    # 1) 블러
    blur = cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    
    # 2) HLS 변환, 밝기 채널
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    l = hls[:,:,1]
    
    # 3) 밝은 영역 추출 (도로 가능성 높은 부분)
    road_mask = cv2.inRange(l, 120, 255)
    
    # 4) 화면 상단 제거 (하늘 등 제외)
    mask_roi = np.zeros_like(road_mask)
    roi_y = int(h*0.6)  # 화면 하단 40%만 ROI로 사용
    mask_roi[roi_y:,:] = 255
    road_mask = cv2.bitwise_and(road_mask, mask_roi)
    
    # 5) 모폴로지로 노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k, iterations=2)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k, iterations=1)
    
    # 6) 가장 큰 컨투어만 남기기
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(road_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", VIDEO_PATH)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        road_mask = detect_road(frame)

        # 시각화 (원본 + 도로 마스크 오버레이)
        color_mask = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
        color_mask = (color_mask>0).astype(np.uint8)*np.array([0,255,255], np.uint8)
        vis = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)

        h, w = vis.shape[:2]
        vis_resized = cv2.resize(vis, (int(w*SCALE), int(h*SCALE)))
        cv2.imshow("Detected Road Only", vis_resized)

        key = cv2.waitKey(1) & 0xFF
        if key==27 or key==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
