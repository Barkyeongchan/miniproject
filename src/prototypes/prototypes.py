import cv2
import numpy as np

# ====================== 설정 ======================
VIDEO_PATH = "../../assets/drive_sample.mp4"  # 분석할 영상 경로
SHOW_BINARY = True          # 작은 창으로 이진 마스크 표시 여부
SAVE_OUTPUT = False         # 결과 저장 여부
OUTPUT_PATH = "lane_output_curve.mp4"  # 저장할 영상 이름

SCALE = 0.6  # 출력 창 크기 비율 (0~1)

# HLS S채널 이진화 임계값
S_MIN, S_MAX = 120, 255

# Sobel X 임계값
SX_MIN, SX_MAX = 25, 255

GAUSS_KSIZE = 5  # 가우시안 블러 커널 크기
MORPH_K = 5      # 모폴로지 커널 크기

HOUGH_THRESH = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 60
# =================================================

# ====================== 함수 정의 ======================

# 1) ROI: 차선 검출된 부분만
def region_of_interest(binary):
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return binary  # 차선 없음 → 전체 반환
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    mask = np.zeros_like(binary)
    mask[y_min:y_max+1, x_min:x_max+1] = 255
    return cv2.bitwise_and(binary, mask)

# 2) 차선 이진화
def threshold_binary(frame):
    blur = cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    # 색상 기반 필터링 (노란색/흰색)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    white_mask  = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))
    color_mask  = cv2.bitwise_or(yellow_mask, white_mask)

    # HLS S채널
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    s_bin = cv2.inRange(s, S_MIN, S_MAX)

    # Sobel X
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sx = cv2.convertScaleAbs(sx)
    sx_bin = cv2.inRange(sx, SX_MIN, SX_MAX)

    # 세 마스크 합치기
    combined = cv2.bitwise_or(color_mask, s_bin)
    combined = cv2.bitwise_or(combined, sx_bin)

    # 모폴로지로 노이즈 제거
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)

    # ROI: 차선 검출된 부분만
    roi_binary = region_of_interest(cleaned)
    return roi_binary

# 3) Hough 직선 검출
def detect_lines(binary):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

# 4) 좌/우 차선 분리
def separate_left_right(lines, img_shape):
    left, right = [], []
    h, w = img_shape[:2]
    if lines is None:
        return left, right

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2 == x1: 
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.3:  # 거의 수평 제외
            continue
        if slope < 0 and max(x1,x2) < w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope > 0 and min(x1,x2) > w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right

# 5) 차선 점선 이어서 그리기 (1차 직선 fitting)
def fit_and_draw_lane(frame, segments, color, thickness=8):
    if len(segments) == 0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    xs = np.array(xs)
    ys = np.array(ys)

    # 1차 least squares
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]

    h = frame.shape[0]
    y_bottom = ys.max()
    y_top = ys.min()
    x_bottom = int(a*y_bottom + b)
    x_top = int(a*y_top + b)

    cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness, cv2.LINE_AA)
    return (x_bottom, y_bottom, x_top, y_top)

# 6) 이진 마스크 오버레이
def overlay_mask(frame, binary, alpha=0.35):
    color_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    color_mask = (color_mask>0).astype(np.uint8)*np.array([0,255,255], np.uint8)
    blended = cv2.addWeighted(frame, 1.0, color_mask, alpha, 0)
    return blended

# ====================== 메인 ======================
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
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 차선 이진화 + ROI
        binary = threshold_binary(frame)

        # Hough 직선 검출
        lines = detect_lines(binary)
        left, right = separate_left_right(lines, frame.shape)

        # 시각화
        vis = overlay_mask(frame.copy(), binary) if SHOW_BINARY else frame.copy()
        fit_and_draw_lane(vis, left,  (0, 255, 0), thickness=10)
        fit_and_draw_lane(vis, right, (0, 150, 255), thickness=10)

        h, w = vis.shape[:2]
        txt = f"L:{len(left):02d} seg  R:{len(right):02d} seg"
        cv2.putText(vis, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30,220,30), 2, cv2.LINE_AA)

        if SHOW_BINARY:
            small = cv2.resize(binary, (w//3, h//3))
            vis[10:10+small.shape[0], w - 10 - small.shape[1]: w-10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (w - 10 - small.shape[1], 10), (w-10, 10+small.shape[0]), (0,0,0), 2)

        vis_resized = cv2.resize(vis, (int(w*SCALE), int(h*SCALE)))
        cv2.imshow("Lane Detection (Curve Fitting)", vis_resized)

        if writer is not None:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
