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
        (int(w*0.35), int(h*ROI_TOP_Y)),
        (int(w*0.65), int(h*ROI_TOP_Y)),
        (int(w*0.9), int(h*ROI_BOTTOM_Y))
    ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))
    return cv2.bitwise_and(img, mask)

# ---------- RANSAC 기반 곡선 ----------
def fit_lane_curve_ransac(segments, degree=2):
    if len(segments) == 0:
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
    y_vals = np.linspace(y_bottom, y_top, abs(y_bottom - y_top)+1)
    x_vals = model.predict(y_vals.reshape(-1,1)).astype(int)
    return x_vals, y_vals

# ---------- 투시 변환 ----------
def get_perspective_transform(frame):
    h, w = frame.shape[:2]
    src_pts = np.float32([
        [int(w*0.1), int(h*ROI_BOTTOM_Y)],
        [int(w*0.35), int(h*ROI_TOP_Y)],
        [int(w*0.65), int(h*ROI_TOP_Y)],
        [int(w*0.9), int(h*ROI_BOTTOM_Y)]
    ])
    dst_pts = np.float32([[0,h],[0,0],[w,0],[w,h]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(frame, M, (w,h))
    return warped, M, Minv

# ---------- 버드아이뷰에서 차선 검출 ----------
def detect_lanes_birdeye(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur, 50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=60)
    left, right = [], []
    h, w = warped.shape[:2]
    if lines is None:
        return left, right, edges
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2==x1: continue
        slope = (y2-y1)/(x2-x1+1e-6)
        if abs(slope)<0.3 or abs(slope)>10: continue
        if slope<0 and max(x1,x2)<w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope>0 and min(x1,x2)>w*0.45:
            right.append((x1,y1,x2,y2))
    return left, right, edges

# ---------- curve 오버레이 ----------
def draw_lane_overlay(frame, left_model, right_model, Minv):
    h, w = frame.shape[:2]
    y_bottom = 0
    y_top = h-1
    overlay = np.zeros_like(frame)
    if left_model is not None and right_model is not None:
        left_x, y_vals = get_lane_points(left_model, y_bottom, y_top)
        right_x, _ = get_lane_points(right_model, y_bottom, y_top)
        pts_left = np.vstack([left_x, y_vals]).T
        pts_right = np.vstack([right_x, y_vals]).T[::-1]
        pts = np.vstack([pts_left, pts_right])
        pts = np.array([pts], dtype=np.int32)
        pts_orig = cv2.perspectiveTransform(pts.reshape(-1,1,2).astype(np.float32), Minv)
        cv2.fillPoly(overlay, [pts_orig.astype(int)], (180,255,255))
    result = cv2.addWeighted(frame,1.0,overlay,0.4,0)
    return result

# ---------- 점선 차선 박스 오버레이 ----------
def draw_lane_boxes(frame, left_segments, right_segments, Minv):
    overlay = np.zeros_like(frame)

    def create_boxes(segments):
        boxes = []
        for x1,y1,x2,y2 in segments:
            top = min(y1,y2)
            bottom = max(y1,y2)
            left = min(x1,x2)-5
            right = max(x1,x2)+5
            boxes.append([left, top, right, bottom])
        return boxes

    left_boxes = create_boxes(left_segments)
    right_boxes = create_boxes(right_segments)

    for box_list in [left_boxes, right_boxes]:
        for left, top, right, bottom in box_list:
            pts = np.array([[[left,top],[right,top],[right,bottom],[left,bottom]]], dtype=np.float32)
            pts_orig = cv2.perspectiveTransform(pts, Minv)
            cv2.fillPoly(overlay, [pts_orig.astype(int)], (0,255,255))

    result = cv2.addWeighted(frame,1.0,overlay,0.4,0)
    return result

# ---------- 메인 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", VIDEO_PATH)
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        warped, M, Minv = get_perspective_transform(frame)
        left, right, edges = detect_lanes_birdeye(warped)
        left_model = fit_lane_curve_ransac(left)
        right_model = fit_lane_curve_ransac(right)

        # --- curve 오버레이 유지 ---
        vis = draw_lane_overlay(frame, left_model, right_model, Minv)
        # --- 점선 차선 박스 추가 ---
        vis = draw_lane_boxes(vis, left, right, Minv)

        # --- 오른쪽에 warp ROI 미리보기 ---
        roi_h, roi_w = warped.shape[:2]
        scale = 0.4
        resized_roi = cv2.resize(warped, (int(roi_w*scale), int(roi_h*scale)))
        x_offset = frame.shape[1] - resized_roi.shape[1] - 10
        y_offset = 10
        vis[y_offset:y_offset+resized_roi.shape[0], x_offset:x_offset+resized_roi.shape[1]] = resized_roi
        cv2.rectangle(vis, (x_offset, y_offset),
                      (x_offset+resized_roi.shape[1], y_offset+resized_roi.shape[0]), (0,0,255), 2)
        cv2.putText(vis, "Warped ROI", (x_offset+5, y_offset+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Lane Detection Smooth Curve", cv2.resize(vis,(int(vis.shape[1]*SCALE), int(vis.shape[0]*SCALE))))
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
