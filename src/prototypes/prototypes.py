import cv2
import numpy as np

# ========= 설정 =========
VIDEO_PATH = "../../assets/drive_sample.mp4"  # 영상 경로
SHOW_BINARY = True          # 이진 마스크 미리보기
SAVE_OUTPUT = False         # 결과 저장 여부
OUTPUT_PATH = "lane_output_fixed_roi.mp4"
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
    """
    관심 영역(ROI)을 정의하고 이미지에 적용합니다.
    """
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
    """
    HLS S채널과 Sobel X를 결합하여 차선이 될만한 부분을 이진화합니다.
    """
    blur = cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), 0)
    # HLS S채널
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    s_bin = cv2.inRange(s, S_MIN, S_MAX)
    # Sobel X
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sx = cv2.convertScaleAbs(sx)
    sx_bin = cv2.inRange(sx, SX_MIN, SX_MAX)
    # 합치기
    combined = cv2.bitwise_or(s_bin, sx_bin)
    combined = region_of_interest(combined)  # ROI 적용
    # 모폴로지
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=2)
    return combined

# ---------- 허프 직선 ----------
def detect_lines(binary):
    """
    이진화된 이미지에서 허프 변환을 사용하여 선분을 검출합니다.
    """
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

# ---------- 좌/우 차선 분리 ----------
def separate_left_right(lines, img_shape):
    """
    기울기와 위치를 기준으로 선분을 좌/우 차선으로 분리합니다.
    중앙 영역의 선분은 무시합니다.
    """
    left, right = [], []
    h, w = img_shape[:2]
    if lines is None:
        return left, right
    for l in lines:
        x1, y1, x2, y2 = l[0]
        if x2==x1:
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        # 기울기가 너무 수평이거나 수직인 선분은 제외
        if abs(slope) < 0.3 or abs(slope) > 10:
            continue
        # 수정된 부분: 중앙 영역(40% ~ 60%)에 있는 선분은 제외
        # 왼쪽 차선은 음수 기울기, 오른쪽 차선은 양수 기울기
        if slope < 0 and max(x1,x2) < w*0.4:
            left.append((x1,y1,x2,y2))
        elif slope > 0 and min(x1,x2) > w*0.6:
            right.append((x1,y1,x2,y2))
    return left, right

# ---------- 점선 자연 연결 ----------
def fit_and_draw_lane(frame, segments, color, thickness=8):
    """
    선분들을 회귀 분석하여 하나의 차선으로 연결하고 그립니다.
    """
    if len(segments) == 0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1,x2]
        ys += [y1,y2]
    xs = np.array(xs)
    ys = np.array(ys)
    # y에 따른 x 회귀
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]
    h = frame.shape[0]
    y_bottom = int(h*ROI_BOTTOM_Y)
    y_top = int(h*ROI_TOP_Y)
    x_bottom = int(a*y_bottom + b)
    x_top    = int(a*y_top + b)
    cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness, cv2.LINE_AA)
    return (x_bottom, y_bottom, x_top, y_top)

# ---------- 마스크 오버레이 ----------
def overlay_mask(frame, binary, alpha=0.35):
    """
    차선 마스크를 원본 영상에 투명하게 겹쳐서 보여줍니다.
    """
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
        cv2.imshow("Lane Detection (Fixed ROI)", vis_resized)

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
