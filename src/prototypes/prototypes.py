import cv2
import numpy as np

# ====================== 설정 ======================
VIDEO_PATH = "../../assets/drive_sample.mp4"  # 분석할 영상 경로
SHOW_BINARY = True          # 작은 창으로 이진 마스크 표시 여부
SAVE_OUTPUT = False         # 결과 저장 여부
OUTPUT_PATH = "lane_output.mp4"  # 저장할 영상 이름

SCALE = 0.6  # 출력 창 크기 비율 (0~1)

# HLS S채널 이진화 임계값
S_MIN, S_MAX = 120, 255

# Sobel X(수평 방향 경계) 임계값
SX_MIN, SX_MAX = 25, 255

GAUSS_KSIZE = 5  # 가우시안 블러 커널 크기
MORPH_K = 5      # 모폴로지 커널 크기

# Hough 변환 파라미터
HOUGH_THRESH = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 60

# ROI 비율 (사다리꼴)
ROI_BOTTOM_Y = 0.95   # 하단
ROI_TOP_Y    = 0.62   # 상단
ROI_LEFT_X   = 0.10   # 왼쪽 끝
ROI_RIGHT_X  = 0.90   # 오른쪽 끝
ROI_TOP_LEFT_X  = 0.42
ROI_TOP_RIGHT_X = 0.58
# =================================================

# ====================== 함수 정의 ======================

# 1) ROI (관심영역) 사다리꼴 마스크
def region_of_interest(img):
    h, w = img.shape[:2]
    # 사다리꼴 좌표
    pts = np.array([[ 
        (int(w*ROI_LEFT_X),  int(h*ROI_BOTTOM_Y)),
        (int(w*ROI_TOP_LEFT_X),  int(h*ROI_TOP_Y)),
        (int(w*ROI_TOP_RIGHT_X), int(h*ROI_TOP_Y)),
        (int(w*ROI_RIGHT_X), int(h*ROI_BOTTOM_Y))
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))
    # ROI 영역만 남기고 나머지는 0
    return cv2.bitwise_and(img, mask)

# 2) 차선 이진화 (색상 + S채널 + Sobel)
def threshold_binary(frame):
    # 가우시안 블러로 노이즈 제거
    blur = cv2.GaussianBlur(frame, (GAUSS_KSIZE, GAUSS_KSIZE), 0)

    # HSV 변환 후 노란색 + 흰색 마스크
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))  # 노란색
    white_mask  = cv2.inRange(hsv, (0, 0, 200), (255, 30, 255))    # 흰색
    color_mask  = cv2.bitwise_or(yellow_mask, white_mask)

    # HLS S채널 마스크
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    s_bin = cv2.inRange(s, S_MIN, S_MAX)

    # Sobel X(수평 에지)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sx = cv2.convertScaleAbs(sx)
    sx_bin = cv2.inRange(sx, SX_MIN, SX_MAX)

    # 세 가지 마스크 통합
    combined = cv2.bitwise_or(color_mask, s_bin)
    combined = cv2.bitwise_or(combined, sx_bin)

    # ROI 적용
    combined_roi = region_of_interest(combined)

    # 모폴로지(닫기)로 노이즈 제거 및 선 연속화
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_K, MORPH_K))
    cleaned = cv2.morphologyEx(combined_roi, cv2.MORPH_CLOSE, k, iterations=2)
    return cleaned

# 3) 차선 내부 글씨/횡단보도 제거
def remove_text_like_regions(binary):
    # 외곽선(contour) 검출
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)
        area = w * h
        # 넓게 퍼진 수평 영역은 제거 (횡단보도, 글씨)
        if area > 500 and aspect_ratio > 2.5:
            continue
        # 나머지 영역은 유지
        cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask

# 4) Hough 라인 검출
def detect_lines(binary):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HOUGH_THRESH,
                            minLineLength=HOUGH_MIN_LINE_LEN,
                            maxLineGap=HOUGH_MAX_LINE_GAP)
    return lines

# 5) 좌/우 차선 분리
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
        # 너무 수평/급경사 제외
        if abs(slope) < 0.4 or abs(slope) > 10:
            continue
        # 왼쪽/오른쪽 구분
        if slope < 0 and max(x1,x2) < w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope > 0 and min(x1,x2) > w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right

# 6) 직선 적합 후 차선 그리기
def fit_and_draw_lane(frame, segments, color, thickness=8):
    if len(segments) == 0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    xs = np.array(xs); ys = np.array(ys)

    # y = ax + b 형태가 아닌 x = a*y + b 형태로 fitting (화면 좌표 기준)
    A = np.vstack([ys, np.ones_like(ys)]).T
    a, b = np.linalg.lstsq(A, xs, rcond=None)[0]

    h = frame.shape[0]
    y_bottom = int(h*ROI_BOTTOM_Y)
    y_top    = int(h*ROI_TOP_Y)
    x_bottom = int(a*y_bottom + b)
    x_top    = int(a*y_top + b)

    cv2.line(frame, (x_bottom, y_bottom), (x_top, y_top), color, thickness, cv2.LINE_AA)
    return (x_bottom, y_bottom, x_top, y_top)

# 7) 이진 마스크 오버레이
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

        # 1) 차선 이진화
        binary = threshold_binary(frame)
        # 2) 글씨/횡단보도 제거
        binary = remove_text_like_regions(binary)
        # 3) Hough 라인 검출
        lines = detect_lines(binary)
        # 4) 좌/우 차선 분리
        left, right = separate_left_right(lines, frame.shape)

        # 시각화
        vis = overlay_mask(frame.copy(), binary) if SHOW_BINARY else frame.copy()
        left_line  = fit_and_draw_lane(vis, left,  (0, 255, 0), thickness=10)
        right_line = fit_and_draw_lane(vis, right, (0, 150, 255), thickness=10)

        # 좌/우 라인 수 표시
        txt = f"L:{len(left):02d} seg  R:{len(right):02d} seg"
        cv2.putText(vis, txt, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30,220,30), 2, cv2.LINE_AA)

        # 작은 이진 마스크 미리보기
        if SHOW_BINARY:
            small = cv2.resize(binary, (w//3, h//3))
            vis[10:10+small.shape[0], w - 10 - small.shape[1]: w-10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis,
                          (w - 10 - small.shape[1], 10),
                          (w-10, 10+small.shape[0]),
                          (0,0,0), 2)

        # 스케일 다운 후 표시
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
