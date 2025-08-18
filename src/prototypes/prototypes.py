import cv2
import numpy as np

# ========= 설정 =========
VIDEO_PATH = "../../assets/drive_sample.mp4"  # 영상 경로
SHOW_BINARY = True          # 이진 마스크 미리보기
SAVE_OUTPUT = False         # 결과 저장 여부
OUTPUT_PATH = "lane_output_curve_roi.mp4"
SCALE = 0.6                 # 출력 창 크기 비율

# HLS S채널 임계값
S_MIN, S_MAX = 120, 255

# Sobel X 임계값
SX_MIN, SX_MAX = 25, 255
GAUSS_KSIZE = 5
MORPH_K = 5
HOUGH_THRESH = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 60

# ROI 비율 (하단 사다리꼴)
ROI_BOTTOM_Y = 0.95
ROI_TOP_Y    = 0.6
ROI_LEFT_X   = 0.1
ROI_RIGHT_X  = 0.9
ROI_TOP_LEFT_X  = 0.45
ROI_TOP_RIGHT_X = 0.55

# ---------- ROI 마스크 ----------
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

# ---------- 차선 이진화 ----------
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
    combined = region_of_interest(combined)  # ROI 적용
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    return combined

# ---------- 허프 직선 ----------
def detect_lines(binary):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

# ---------- 좌/우 차선 분리 ----------
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
        if abs(slope) < 0.3 or abs(slope) > 10:
            continue
        if slope < 0 and max(x1,x2) < w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope > 0 and min(x1,x2) > w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right

# ---------- 점선 자연 연결 (곡선) ----------
def fit_and_draw_lane(frame, segments, color, thickness=8, poly_order=2):
    if len(segments) == 0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1,x2]
        ys += [y1,y2]
    xs = np.array(xs)
    ys = np.array(ys)
    coeffs = np.polyfit(ys, xs, poly_order)  # x = f(y)
    h = frame.shape[0]
    y_bottom = int(h*ROI_BOTTOM_Y)
    y_top    = int(h*ROI_TOP_Y)
    y_vals = np.linspace(y_bottom, y_top, num=(y_bottom-y_top+1))
    x_vals = np.polyval(coeffs, y_vals).astype(int)
    for i in range(len(y_vals)-1):
        cv2.line(frame, (x_vals[i], int(y_vals[i])), (x_vals[i+1], int(y_vals[i+1])), color, thickness, cv2.LINE_AA)
    return (x_vals[0], y_bottom, x_vals[-1], y_top)

# ---------- 마스크 오버레이 ----------
def overlay_mask(frame, binary, alpha=0.35):
    color_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    color_mask = (color_mask>0).astype(np.uint8) * np.array([0,255,255], np.uint8)
    return cv2.addWeighted(frame, 1.0, color_mask, alpha, 0)

# ---------- 메인 ----------
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lane_mask = threshold_binary(frame)
        lines = detect_lines(lane_mask)
        left, right = separate_left_right(lines, frame.shape)

        vis = overlay_mask(frame.copy(), lane_mask) if SHOW_BINARY else frame.copy()
        fit_and_draw_lane(vis, left,  (0,255,0), thickness=10)
        fit_and_draw_lane(vis, right, (0,150,255), thickness=10)

        h, w = vis.shape[:2]
        # 작은 창 미리보기
        if SHOW_BINARY:
            small = cv2.resize(lane_mask, (w//3, h//3))
            vis[10:10+small.shape[0], w - 10 - small.shape[1]: w-10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (w - 10 - small.shape[1], 10),
                                 (w-10, 10+small.shape[0]), (0,0,0), 2)

        vis_resized = cv2.resize(vis, (int(w*SCALE), int(h*SCALE)))
        cv2.imshow("Lane Detection (Curve ROI)", vis_resized)

        if writer is not None:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key==27 or key==ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
