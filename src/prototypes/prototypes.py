import cv2
import numpy as np
import time

# ========= 설정 =========
VIDEO_PATH = "../../assets/drive_sample.mp4"  # 영상 경로
SAVE_OUTPUT = False
OUTPUT_PATH = "fixed_30fps_output.mp4"
SCALE = 0.6
TRAPEZOID_TOP_Y = 0.6
TRAPEZOID_BOTTOM_Y = 1.0
TRAPEZOID_TOP_WIDTH = 30
TRAPEZOID_BOTTOM_WIDTH = 600

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

def visualize_only(frame, trapezoid_pts):
    overlay = np.zeros_like(frame, np.uint8)
    cv2.fillPoly(overlay, trapezoid_pts, (0, 255, 0))
    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", VIDEO_PATH)
        return

    target_fps = 30
    frame_time = 1.0 / target_fps
    next_frame_time = time.time()

    writer = None
    if SAVE_OUTPUT:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, target_fps, (w, h))

    ret, frame = cap.read()
    if not ret:
        print("비디오 첫 프레임을 읽을 수 없습니다.")
        return
    trapezoid_points = calculate_fixed_trapezoid(frame.shape)

    # --- FPS 측정용 ---
    frame_counter = 0
    fps_timer = time.time()
    actual_fps = 0.0

    while True:
        # --- 다음 프레임 표시할 시간까지 대기 ---
        sleep_time = next_frame_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

        next_frame_time += frame_time  # 다음 목표 시간 업데이트

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = visualize_only(frame, trapezoid_points)

        # --- 실제 FPS 계산 ---
        frame_counter += 1
        if time.time() - fps_timer >= 1.0:
            actual_fps = frame_counter / (time.time() - fps_timer)
            frame_counter = 0
            fps_timer = time.time()

        # FPS 표시 (왼쪽 상단)
        cv2.putText(processed_frame,
                    f"FPS: {actual_fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        vis_resized = cv2.resize(processed_frame,
                                 (int(processed_frame.shape[1] * SCALE),
                                  int(processed_frame.shape[0] * SCALE)))
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
