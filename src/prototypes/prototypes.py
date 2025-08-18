import cv2
import numpy as np

# ===== 설정 =====
VIDEO_PATH = "../../assets/drive_sample.mp4"
SHOW_BINARY = True        # 이진 마스크 오버레이
SAVE_OUTPUT = False
OUTPUT_PATH = "lane_output_auto.mp4"
SCALE = 0.6

# HLS S채널 + Sobel X
S_MIN, S_MAX = 120, 255
SX_MIN, SX_MAX = 25, 255
GAUSS_KSIZE = 5
MORPH_K = 5
HOUGH_THRESH = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 60

# ===== 도로 영역 자동 검출 =====
def detect_road_roi(frame):
    h, w = frame.shape[:2]
    blur = cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    l = hls[:,:,1]
    road_mask = cv2.inRange(l, 100, 255)

    # 화면 상단 제거 (하늘 등 제외)
    mask_roi = np.zeros_like(road_mask)
    mask_roi[int(h*0.6):, :] = 255
    road_mask = cv2.bitwise_and(road_mask, mask_roi)

    # 모폴로지
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k, iterations=2)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, k, iterations=1)

    # 가장 큰 컨투어만
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(road_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask

# ===== 차선 이진화 =====
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
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    return combined

# ===== Hough 직선 검출 =====
def detect_lines(binary):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

# ===== 좌/우 차선 분리 =====
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
        if abs(slope)<0.4 or abs(slope)>10:
            continue
        if slope < 0 and max(x1,x2) < w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope > 0 and min(x1,x2) > w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right

# ===== 점선 이어서 차선 그리기 =====
def fit_and_draw_lane(frame, segments, color, thickness=8):
    if len(segments)==0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1,x2]; ys += [y1,y2]
    xs = np.array(xs); ys = np.array(ys)
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
    h = frame.shape[0]
    y_bottom = h-1
    y_top = int(h*0.6)
    x_bottom = int(a*y_bottom + b)
    x_top    = int(a*y_top + b)
    cv2.line(frame, (x_bottom,y_bottom), (x_top,y_top), color, thickness, cv2.LINE_AA)
    return (x_bottom,y_bottom,x_top,y_top)

# ===== 오버레이 =====
def overlay_mask(frame, binary, alpha=0.35):
    color_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    color_mask = (color_mask>0).astype(np.uint8)*np.array([0,255,255], np.uint8)
    return cv2.addWeighted(frame, 1.0, color_mask, alpha, 0)

# ===== 메인 =====
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오 열기 실패:", VIDEO_PATH)
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

        road_mask = detect_road_roi(frame)
        lane_mask = threshold_binary(frame)
        lane_mask = cv2.bitwise_and(lane_mask, road_mask)  # 도로 위 차선만

        lines = detect_lines(lane_mask)
        left, right = separate_left_right(lines, frame.shape)

        vis = overlay_mask(frame.copy(), lane_mask) if SHOW_BINARY else frame.copy()
        fit_and_draw_lane(vis, left,  (0,255,0), thickness=10)
        fit_and_draw_lane(vis, right, (0,150,255), thickness=10)

        # 작은 창으로 원본 + 이진화 확인
        h, w = vis.shape[:2]
        if SHOW_BINARY:
            small = cv2.resize(lane_mask, (w//3, h//3))
            vis[10:10+small.shape[0], w-10-small.shape[1]:w-10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (w-10-small.shape[1],10), (w-10,10+small.shape[0]), (0,0,0),2)

        vis_resized = cv2.resize(vis, (int(w*SCALE), int(h*SCALE)))
        cv2.imshow("Lane Detection (Auto ROI)", vis_resized)

        if writer:
            writer.write(vis)

        key = cv2.waitKey(1) & 0xFF
        if key==27 or key==ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
