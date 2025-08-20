import cv2
import numpy as np

# ========= 설정 =========
VIDEO_PATH = "../../assets/drive_sample.mp4"  # 영상 경로
SAVE_OUTPUT = False         # 결과 저장 여부
OUTPUT_PATH = "fixed_central_roi_output.mp4"
SCALE = 0.6                 # 출력 창 크기 비율

# 고정 사다리꼴의 기본 설정값
TRAPEZOID_TOP_Y = 0.6
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 100   # 위쪽 너비 (픽셀 단위)
TRAPEZOID_BOTTOM_WIDTH = 600 # 아래쪽 너비 (픽셀 단위)

# ---------- 고정된 사다리꼴 좌표 계산 ----------
def calculate_fixed_trapezoid(frame_shape):
    """
    화면 중앙에 고정된 사다리꼴 ROI의 좌표를 계산합니다.
    """
    h, w = frame_shape[:2]
    
    # 화면 중앙을 기준으로 사다리꼴 좌표 계산
    center_x = w // 2

    top_y = int(h * TRAPEZOID_TOP_Y)
    bottom_y = int(h * TRAPEZOID_BOTTOM_Y)

    # 중심점을 기준으로 좌우 좌표를 설정
    top_left_x = center_x - TRAPEZOID_TOP_WIDTH // 2
    top_right_x = center_x + TRAPEZOID_TOP_WIDTH // 2
    bottom_left_x = center_x - TRAPEZOID_BOTTOM_WIDTH // 2
    bottom_right_x = center_x + TRAPEZOID_BOTTOM_WIDTH // 2

    # 좌표를 numpy 배열로 저장
    trapezoid_pts = np.array([[
        (bottom_left_x, bottom_y),
        (top_left_x, top_y),
        (top_right_x, top_y),
        (bottom_right_x, bottom_y)
    ]], dtype=np.int32)

    return trapezoid_pts

# ---------- ROI 시각화만 수행 ----------
def visualize_only(frame, trapezoid_pts):
    """
    고정된 사다리꼴을 시각화합니다.
    이전의 텍스트와 원 표시 기능을 제거했습니다.
    """
    overlay = np.zeros_like(frame, np.uint8)
    
    # 사다리꼴 ROI를 초록색으로 채우기
    cv2.fillPoly(overlay, trapezoid_pts, (0, 255, 0))
    result = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)
    
    return result

# ---------- 메인 루프 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", VIDEO_PATH)
        return

    writer = None
    if SAVE_OUTPUT:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w,h))

    # 고정된 ROI는 한 번만 계산합니다.
    ret, frame = cap.read()
    if not ret:
        print("비디오 첫 프레임을 읽을 수 없습니다.")
        return
        
    trapezoid_points = calculate_fixed_trapezoid(frame.shape)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 고정된 사다리꼴을 시각화만 합니다.
        processed_frame = visualize_only(frame, trapezoid_points)
                             
        vis_resized = cv2.resize(processed_frame, (int(processed_frame.shape[1]*SCALE), int(processed_frame.shape[0]*SCALE)))
        cv2.imshow("Fixed Central ROI", vis_resized)

        if writer is not None:
            writer.write(processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key==27 or key==ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
