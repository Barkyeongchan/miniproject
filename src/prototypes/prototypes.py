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
ROI_BOTTOM_Y = 1
ROI_TOP_Y    = 0.6
ROI_LEFT_X   = 0.25
ROI_RIGHT_X  = 0.85
ROI_TOP_LEFT_X  = 0.45
ROI_TOP_RIGHT_X = 0.55

# 새로운 설정: 하단 중앙 노이즈를 무시하기 위한 사다리꼴 제외 영역
EXCLUSION_BOTTOM_LEFT_X = 0.3
EXCLUSION_BOTTOM_RIGHT_X = 0.8
EXCLUSION_TOP_LEFT_X = 0.45
EXCLUSION_TOP_RIGHT_X = 0.55
EXCLUSION_TOP_Y = 0.85
EXCLUSION_BOTTOM_Y = 1.0

# 추가 설정: 두 번째 중앙 노이즈 제외 영역 (삼각형 형태)
# 아래쪽 너비는 첫 번째 제외 영역의 상단과 동일하게 설정
EXCLUSION_2_BOTTOM_LEFT_X = EXCLUSION_TOP_LEFT_X
EXCLUSION_2_BOTTOM_RIGHT_X = EXCLUSION_TOP_RIGHT_X
# 위쪽 꼭짓점의 X좌표는 중앙으로 수렴하여 삼각형을 만듦
EXCLUSION_2_TOP_X = (EXCLUSION_2_BOTTOM_LEFT_X + EXCLUSION_2_BOTTOM_RIGHT_X) / 2
# 아래쪽 경계는 첫 번째 제외 영역의 상단과 동일
EXCLUSION_2_BOTTOM_Y = EXCLUSION_TOP_Y
# 위쪽 경계 (임의로 설정)
EXCLUSION_2_TOP_Y = 0.4


# ---------- ROI 마스크 생성 ----------
def region_of_interest(img):
    """
    관심 영역(ROI)을 정의하고 이미지에 적용합니다.
    하단 중앙의 특정 영역들을 제외하여 노이즈를 제거합니다.
    """
    h, w = img.shape[:2]
    
    # 1. 기본 차선 ROI 마스크 생성 (사다리꼴)
    pts = np.array([[ 
        (int(w*ROI_LEFT_X),  int(h*ROI_BOTTOM_Y)),
        (int(w*ROI_TOP_LEFT_X),  int(h*ROI_TOP_Y)),
        (int(w*ROI_TOP_RIGHT_X), int(h*ROI_TOP_Y)),
        (int(w*ROI_RIGHT_X), int(h*ROI_BOTTOM_Y))
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))

    # 2. 첫 번째 하단 중앙 노이즈 제외 영역 마스크 생성 (사다리꼴)
    exclusion_pts_1 = np.array([[
        (int(w*EXCLUSION_BOTTOM_LEFT_X),  int(h*EXCLUSION_BOTTOM_Y)),
        (int(w*EXCLUSION_TOP_LEFT_X),  int(h*EXCLUSION_TOP_Y)),
        (int(w*EXCLUSION_TOP_RIGHT_X), int(h*EXCLUSION_TOP_Y)),
        (int(w*EXCLUSION_BOTTOM_RIGHT_X), int(h*EXCLUSION_BOTTOM_Y))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, exclusion_pts_1, 0)

    # 3. 두 번째 중앙 노이즈 제외 영역 마스크 생성 (삼각형)
    exclusion_pts_2 = np.array([[
        (int(w*EXCLUSION_2_BOTTOM_LEFT_X), int(h*EXCLUSION_2_BOTTOM_Y)),
        (int(w*EXCLUSION_2_BOTTOM_RIGHT_X), int(h*EXCLUSION_2_BOTTOM_Y)),
        (int(w*EXCLUSION_2_TOP_X), int(h*EXCLUSION_2_TOP_Y))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, exclusion_pts_2, 0)
    
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
        # 중앙 영역(40% ~ 60%)에 있는 선분은 무시
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

# ---------- ROI 및 제외 영역 시각화 ----------
def visualize_roi(frame):
    """
    ROI와 제외 영역을 색깔 있는 투명한 폴리곤으로 시각화합니다.
    """
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame, np.uint8)
    
    # 1. ROI 영역을 초록색으로 채우기
    roi_pts = np.array([[ 
        (int(w*ROI_LEFT_X),  int(h*ROI_BOTTOM_Y)),
        (int(w*ROI_TOP_LEFT_X),  int(h*ROI_TOP_Y)),
        (int(w*ROI_TOP_RIGHT_X), int(h*ROI_TOP_Y)),
        (int(w*ROI_RIGHT_X), int(h*ROI_BOTTOM_Y))
    ]], dtype=np.int32)
    cv2.fillPoly(overlay, roi_pts, (0, 255, 0)) # 초록색

    # 2. 첫 번째 제외 영역을 빨간색으로 채우기
    exclusion_pts_1 = np.array([[
        (int(w*EXCLUSION_BOTTOM_LEFT_X),  int(h*EXCLUSION_BOTTOM_Y)),
        (int(w*EXCLUSION_TOP_LEFT_X),  int(h*EXCLUSION_TOP_Y)),
        (int(w*EXCLUSION_TOP_RIGHT_X), int(h*EXCLUSION_TOP_Y)),
        (int(w*EXCLUSION_BOTTOM_RIGHT_X), int(h*EXCLUSION_BOTTOM_Y))
    ]], dtype=np.int32)
    cv2.fillPoly(overlay, exclusion_pts_1, (0, 0, 255)) # 빨간색

    # 3. 두 번째 제외 영역을 빨간색으로 채우기
    exclusion_pts_2 = np.array([[
        (int(w*EXCLUSION_2_BOTTOM_LEFT_X), int(h*EXCLUSION_2_BOTTOM_Y)),
        (int(w*EXCLUSION_2_BOTTOM_RIGHT_X), int(h*EXCLUSION_2_BOTTOM_Y)),
        (int(w*EXCLUSION_2_TOP_X), int(h*EXCLUSION_2_TOP_Y))
    ]], dtype=np.int32)
    cv2.fillPoly(overlay, exclusion_pts_2, (0, 0, 255)) # 빨간색
    
    # 원본 이미지와 오버레이를 겹쳐서 반환
    return cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

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
        
        # 1. ROI/제외 영역 시각화
        vis = visualize_roi(frame.copy())
        
        # 2. 감지된 차선 그리기
        fit_and_draw_lane(vis, left,  (0,255,0), thickness=10)
        fit_and_draw_lane(vis, right, (0,150,255), thickness=10)

        h, w = vis.shape[:2]
        # 3. 이진 마스크를 작은 창으로 미리보기
        if SHOW_BINARY:
            small = cv2.resize(lane_mask, (w//3, h//3))
            # 이진 마스크를 3채널 컬러 이미지로 변환
            small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            vis[10:10+small_bgr.shape[0], w - 10 - small_bgr.shape[1]: w-10] = small_bgr
            cv2.rectangle(vis, (w - 10 - small_bgr.shape[1], 10),
                                 (w-10, 10+small_bgr.shape[0]), (0,0,0), 2)

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
