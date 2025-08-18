import cv2
import numpy as np

# ========= 설정 =========
VIDEO_PATH = "../../assets/drive_sample.mp4"     # 영상 경로
SHOW_BINARY = True          # 이진 마스크 작은 창 미리보기
SAVE_OUTPUT = False         # 결과 저장 여부
OUTPUT_PATH = "lane_output.mp4"

# 출력 창 크기 비율
SCALE = 0.6  # (예: 0.5 = 절반 크기)

# HLS S채널 이진화 임계값
S_MIN, S_MAX = 120, 255

# Sobel X(수평 방향 에지 강조) 임계값
SX_MIN, SX_MAX = 25, 255
GAUSS_KSIZE = 5
MORPH_K = 5
HOUGH_THRESH = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 60

# ROI 비율 (하단 사다리꼴)
ROI_BOTTOM_Y = 0.95
ROI_TOP_Y    = 0.62
ROI_LEFT_X   = 0.10
ROI_RIGHT_X  = 0.90
ROI_TOP_LEFT_X  = 0.42
ROI_TOP_RIGHT_X = 0.58
# =======================

def region_of_interest(img):
    h, w = img.shape[:2]
    pts = np.array([[
        (int(w*ROI_LEFT_X),  int(h*ROI_BOTTOM_Y)),
        (int(w*ROI_TOP_LEFT_X),  int(h*ROI_TOP_Y)),
        (int(w*ROI_TOP_RIGHT_X), int(h*ROI_TOP_Y)),
        (int(w*ROI_RIGHT_X), int(h*ROI_BOTTOM_Y))
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))
    return cv2.bitwise_and(img, mask)

def threshold_binary(frame):
    blur = cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    s_bin = cv2.inRange(s, S_MIN, S_MAX)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sx = cv2.convertScaleAbs(sx)
    sx_bin = cv2.inRange(sx, SX_MIN, SX_MAX)

    combined = cv2.bitwise_or(s_bin, sx_bin)
    combined_roi = region_of_interest(combined)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    opened = cv2.morphologyEx(combined_roi, cv2.MORPH_OPEN, k, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=1)

    return closed

def detect_lines(binary):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

def separate_left_right(lines, img_shape):
    left, right = [], []
    h, w = img_shape[:2]
    if lines is None:
        return left, right

    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2==x1: 
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.4: 
            continue
        if abs(slope) > 10: 
            continue
        if slope < 0 and max(x1,x2) < w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope > 0 and min(x1,x2) > w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right

def fit_and_draw_lane(frame, segments, color, thickness=8):
    if len(segments) == 0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    xs = np.array(xs); ys = np.array(ys)

    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]

    h = frame.shape[0]
    y_bottom = int(h*ROI_BOTTOM_Y)
    y_top    = int(h*ROI_TOP_Y)
    x_bottom = int(a*y_bottom + b)
    x_top    = int(a*y_top + b)

    cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness, cv2.LINE_AA)
    return (x_bottom, y_bottom, x_top, y_top)

def overlay_mask(frame, binary, alpha=0.35):
    color_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    color_mask = (color_mask>0).astype(np.uint8)*np.array([0,255,255], np.uint8)
    blended = cv2.addWeighted(frame, 1.0, color_mask, alpha, 0)
    return blended

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

        binary = threshold_binary(frame)
        lines = detect_lines(binary)
        left, right = separate_left_right(lines, frame.shape)

        vis = overlay_mask(frame.copy(), binary) if SHOW_BINARY else frame.copy()
        left_line  = fit_and_draw_lane(vis, left,  (0, 255, 0), thickness=10)
        right_line = fit_and_draw_lane(vis, right, (0, 150, 255), thickness=10)

        h, w = vis.shape[:2]
        cv2.line(vis, (w//2, h-40), (w//2, int(h*0.6)), (200,200,200), 1, cv2.LINE_AA)

        txt = f"L:{len(left):02d} seg  R:{len(right):02d} seg"
        cv2.putText(vis, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30,220,30), 2, cv2.LINE_AA)

        if SHOW_BINARY:
            small = cv2.resize(binary, (w//3, h//3))
            vis[10:10+small.shape[0], w - 10 - small.shape[1]: w-10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis,
                          (w - 10 - small.shape[1], 10),
                          (w-10, 10+small.shape[0]),
                          (0,0,0), 2)

        # ===== 스케일 다운 후 표시 =====
        vis_resized = cv2.resize(vis, (int(w*SCALE), int(h*SCALE)))
        cv2.imshow("Lane Detection (Threshold + Hough)", vis_resized)

        if writer is not None:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
