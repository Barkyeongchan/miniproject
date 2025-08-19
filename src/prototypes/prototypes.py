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
    return cv2.bitwise_and(img, mask), pts

# ---------- RANSAC 기반 curve ----------
def fit_lane_curve_ransac(segments, degree=2):
    if len(segments) == 0: return None
    xs, ys = [], []
    for x1,y1,x2,y2 in segments:
        xs += [x1,x2]; ys += [y1,y2]
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) < degree+1: return None
    ys_reshaped = ys.reshape(-1,1)
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(LinearRegression()))
    try: model.fit(ys_reshaped, xs)
    except: return None
    return model

def get_lane_points(model, y_bottom, y_top):
    y_vals = np.linspace(y_bottom, y_top, abs(y_bottom - y_top)+1)
    x_vals = model.predict(y_vals.reshape(-1,1)).astype(int)
    return x_vals, y_vals

# ---------- 투시 변환 ----------
def get_perspective_transform(frame, src_pts):
    h, w = frame.shape[:2]
    dst_pts = np.float32([[0,h],[0,0],[w,0],[w,h]])
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts.astype(np.float32))
    warped = cv2.warpPerspective(frame, M, (w,h))
    return warped, M, Minv

# ---------- 버드아이뷰 차선 검출 ----------
def detect_lanes_birdeye(warped):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=60)
    segments = []
    h, w = warped.shape[:2]
    if lines is None: return segments
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2==x1: continue
        slope = (y2-y1)/(x2-x1+1e-6)
        if abs(slope)<0.3 or abs(slope)>10: continue
        segments.append((x1,y1,x2,y2))
    return segments

# ---------- 점선 패턴 필터 ----------
def filter_dashed_pattern(segments, max_gap=20):
    if not segments: return []
    # 세로 기준 정렬
    segments = sorted(segments, key=lambda b: min(b[1],b[3]))
    filtered = []
    prev_bottom = None
    pattern_lengths = []
    for x1,y1,x2,y2 in segments:
        top = min(y1,y2)
        bottom = max(y1,y2)
        if prev_bottom is None or top - prev_bottom <= max_gap:
            filtered.append((x1,y1,x2,y2))
            pattern_lengths.append(bottom - top)
            prev_bottom = bottom
        else:
            # gap 너무 크면 패턴 끊김, 다시 시작
            prev_bottom = bottom
    # 최소한 반복 패턴으로 판단: 평균 길이 이상 또는 2개 이상 연속
    if len(filtered) < 2: return []
    return filtered

# ---------- 박스 연결 및 필터 ----------
def merge_and_filter_boxes(boxes):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[1])
    merged = []
    current = boxes[0]
    for b in boxes[1:]:
        extended_current = [current[0], current[1]-5, current[2], current[3]+5]
        if not (b[0] > extended_current[2] or b[2] < extended_current[0] or b[1] > extended_current[3] or b[3] < extended_current[1]):
            current = [min(current[0],b[0]), min(current[1],b[1]),
                       max(current[2],b[2]), max(current[3],b[3])]
            if current[3]-current[1] > 3*(b[3]-b[1]):
                current = b
        else:
            merged.append(current)
            current = b
    merged.append(current)
    return merged

# ---------- 버드아이뷰 박스 표시 ----------
def draw_lane_boxes_birdeye_merged_filtered(warped, segments):
    boxes = []
    for x1,y1,x2,y2 in segments:
        top = min(y1,y2); bottom = max(y1,y2)
        left = min(x1,x2)-5; right = max(x1,x2)+5
        boxes.append([left,top,right,bottom])
    merged_boxes = merge_and_filter_boxes(boxes)
    overlay = np.zeros_like(warped)
    for left,top,right,bottom in merged_boxes:
        cv2.rectangle(overlay,(left,top),(right,bottom),(0,0,255),3)
    result = cv2.addWeighted(warped,1.0,overlay,0.4,0)
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

        roi_frame, roi_pts = region_of_interest(frame)
        warped_roi, M, Minv = get_perspective_transform(frame, roi_pts[0])
        segments = detect_lanes_birdeye(warped_roi)
        segments = filter_dashed_pattern(segments)  # 점선 패턴 필터 적용

        # --- 버드아이뷰 박스 통합+필터 ---
        vis_birdeye = draw_lane_boxes_birdeye_merged_filtered(warped_roi, segments)

        # --- 오른쪽 상단 축소 표시 ---
        roi_h, roi_w = vis_birdeye.shape[:2]
        scale = 0.4
        resized_roi = cv2.resize(vis_birdeye,(int(roi_w*scale), int(roi_h*scale)))
        x_offset = frame.shape[1] - resized_roi.shape[1] - 10
        y_offset = 10
        frame[y_offset:y_offset+resized_roi.shape[0], x_offset:x_offset+resized_roi.shape[1]] = resized_roi
        cv2.rectangle(frame,(x_offset,y_offset),(x_offset+resized_roi.shape[1],y_offset+resized_roi.shape[0]),(0,0,255),2)
        cv2.putText(frame,"Warped ROI",(x_offset+5,y_offset+25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        cv2.imshow("Lane Detection BirdEye + Pattern Filtered Boxes", cv2.resize(frame,(int(frame.shape[1]*SCALE),int(frame.shape[0]*SCALE))))
        if cv2.waitKey(1) & 0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
