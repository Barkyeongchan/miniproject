import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ========= 설정 =========
VIDEO_PATH = "../../assets/drive_sample.mp4"
SCALE = 0.6
ROI_BOTTOM_Y = 0.95
ROI_TOP_Y    = 0.6

# ---------- ROI 마스크 ----------
def region_of_interest(img):
    h, w = img.shape[:2]
    pts = np.array([[ 
        (int(w*0.1), int(h*ROI_BOTTOM_Y)),
        (int(w*0.35), int(h*ROI_TOP_Y)),   # 좌측 상단 0.35
        (int(w*0.65), int(h*ROI_TOP_Y)),   # 우측 상단 0.65
        (int(w*0.9), int(h*ROI_BOTTOM_Y))
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))
    return cv2.bitwise_and(img, mask)

# ---------- 차선 이진화 ----------
def threshold_binary(frame):
    blur = cv2.GaussianBlur(frame, (5,5),0)
    hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    s_bin = cv2.inRange(s, 120, 255)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_16S, 1,0, ksize=3)
    sx = cv2.convertScaleAbs(sx)
    sx_bin = cv2.inRange(sx, 25, 255)
    combined = cv2.bitwise_or(s_bin, sx_bin)
    combined = region_of_interest(combined)
    return combined

# ---------- 허프 직선 ----------
def detect_lines(binary):
    edges = cv2.Canny(binary,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=60)
    return lines

# ---------- 좌/우 차선 분리 ----------
def separate_left_right(lines, img_shape):
    left, right = [], []
    h,w = img_shape[:2]
    if lines is None:
        return left, right
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2==x1: continue
        slope = (y2-y1)/(x2-x1+1e-6)
        if abs(slope)<0.3 or abs(slope)>10: continue
        if slope<0 and max(x1,x2)<w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope>0 and min(x1,x2)>w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right

# ---------- RANSAC 기반 곡선 ----------
def fit_lane_curve_ransac(segments, degree=2):
    if len(segments)==0:
        return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1,x2]
        ys += [y1,y2]
    xs = np.array(xs)
    ys = np.array(ys)
    if len(xs) < degree+1:
        return None
    ys_reshaped = ys.reshape(-1,1)
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(LinearRegression()))
    try:
        model.fit(ys_reshaped, xs)
    except:
        return None
    return model

def get_lane_points(model, y_bottom, y_top):
    length = abs(y_bottom - y_top) + 1
    y_vals = np.linspace(y_bottom, y_top, length)
    x_vals = model.predict(y_vals.reshape(-1,1)).astype(int)
    return x_vals, y_vals

# ---------- ROI 투시 보정 후 좁은 ROI 재설정 ----------
def calibrate_and_crop_lane(frame, left_segments, right_segments):
    h, w = frame.shape[:2]
    
    # 1. 원래 ROI 좌표 (상단 좌우 0.35/0.65)
    src_pts = np.float32([
        [int(w*0.1), int(h*ROI_BOTTOM_Y)],
        [int(w*0.35), int(h*ROI_TOP_Y)],
        [int(w*0.65), int(h*ROI_TOP_Y)],
        [int(w*0.9), int(h*ROI_BOTTOM_Y)]
    ])
    dst_pts = np.float32([[0,h],[0,0],[w,0],[w,h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (w,h))

    # 2. 좌우 차선 x좌표 평균
    left_xs = [x for seg in left_segments for x in [seg[0], seg[2]]]
    right_xs = [x for seg in right_segments for x in [seg[0], seg[2]]]
    if len(left_xs)==0 or len(right_xs)==0:
        return warped, M, 0
    left_mean = int(np.mean(left_xs))
    right_mean = int(np.mean(right_xs))
    
    # 3. 차선 폭 기준으로 좌우 3배 폭만 ROI 지정
    lane_width = right_mean - left_mean
    center_x = (left_mean+right_mean)//2
    x1 = max(center_x - lane_width*3, 0)
    x2 = min(center_x + lane_width*3, w)
    cropped = warped[:, x1:x2]
    
    return cropped, M, x1

# ---------- 하늘색 영역 시각화 ----------
def draw_lane_overlay_curve(frame, left_segments, right_segments, prev_left_curve, prev_right_curve):
    h = frame.shape[0]
    y_bottom = int(h*ROI_BOTTOM_Y)
    y_top    = int(h*ROI_TOP_Y)

    # 투시보정 후 좁은 ROI
    cropped_roi, M, roi_x1 = calibrate_and_crop_lane(frame, left_segments, right_segments)

    left_model = fit_lane_curve_ransac(left_segments) or prev_left_curve
    right_model = fit_lane_curve_ransac(right_segments) or prev_right_curve

    if left_model is None or right_model is None:
        return frame, prev_left_curve, prev_right_curve

    left_x, y_vals = get_lane_points(left_model, y_bottom, y_top)
    right_x, _     = get_lane_points(right_model, y_bottom, y_top)

    pts_left = np.vstack([left_x, y_vals]).T
    pts_right = np.vstack([right_x, y_vals]).T[::-1]
    pts = np.vstack([pts_left, pts_right])
    pts = np.array([pts], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, pts, (180,255,255))
    result = cv2.addWeighted(frame,1.0,overlay,0.4,0)

    # --- 투시 보정된 ROI 오버레이 ---
    # 오른쪽에 작게 붙이기
    roi_h, roi_w = cropped_roi.shape[:2]
    scale = 0.4  # 40% 크기로 축소
    resized_roi = cv2.resize(cropped_roi, (int(roi_w*scale), int(roi_h*scale)))
    x_offset = frame.shape[1] - resized_roi.shape[1] - 10
    y_offset = 10
    result[y_offset:y_offset+resized_roi.shape[0], x_offset:x_offset+resized_roi.shape[1]] = resized_roi
    cv2.rectangle(result, (x_offset, y_offset),
                  (x_offset+resized_roi.shape[1], y_offset+resized_roi.shape[0]),
                  (0,0,255), 2)
    cv2.putText(result, "Warped ROI", (x_offset+5, y_offset+25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # 기존 미리보기 박스
    lane_mask = threshold_binary(frame)
    small = cv2.resize(lane_mask, (frame.shape[1]//4, frame.shape[0]//4))
    result[10:10+small.shape[0], 10:10+small.shape[1]] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result, (10,10), (10+small.shape[1], 10+small.shape[0]), (0,0,0),2)

    return result, left_model, right_model


# ---------- 메인 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", VIDEO_PATH)
        return

    prev_left_curve, prev_right_curve = None, None

    while True:
        ret, frame = cap.read()
        if not ret: break

        lane_mask = threshold_binary(frame)
        lines = detect_lines(lane_mask)
        left, right = separate_left_right(lines, frame.shape)

        vis, prev_left_curve, prev_right_curve = draw_lane_overlay_curve(
            frame.copy(), left, right, prev_left_curve, prev_right_curve
        )

        cv2.imshow("Lane Detection Smooth Curve",
                   cv2.resize(vis,(int(vis.shape[1]*SCALE), int(vis.shape[0]*SCALE))))
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
