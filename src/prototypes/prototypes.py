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

# 초기 ROI 좌표 (비율)
ROI_POINTS = np.array([
    [0.1, ROI_BOTTOM_Y],
    [0.45, ROI_TOP_Y],
    [0.55, ROI_TOP_Y],
    [0.9, ROI_BOTTOM_Y]
], dtype=np.float32)

# ---------- ROI 마스크 ----------
def region_of_interest(img):
    h, w = img.shape[:2]
    pts = np.array([[ (int(x*w), int(y*h)) for x,y in ROI_POINTS ]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))
    return cv2.bitwise_and(img, mask)

# ---------- 투시 변환 ----------
def warp_roi(frame):
    h, w = frame.shape[:2]
    src = np.array([[x*w, y*h] for x,y in ROI_POINTS], dtype=np.float32)
    dst = np.array([[0,h],[0,0],[w,0],[w,h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    warped = cv2.warpPerspective(frame,M,(w,h))
    return warped, M, Minv

# ---------- 차선 이진화 ----------
def threshold_binary(frame):
    blur = cv2.GaussianBlur(frame,(5,5),0)
    hls = cv2.cvtColor(blur,cv2.COLOR_BGR2HLS)
    s = hls[:,:,2]
    s_bin = cv2.inRange(s,120,255)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=3)
    sx = cv2.convertScaleAbs(sx)
    sx_bin = cv2.inRange(sx,25,255)
    combined = cv2.bitwise_or(s_bin,sx_bin)
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
    if lines is None: return left,right
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2==x1: continue
        slope = (y2-y1)/(x2-x1+1e-6)
        if abs(slope)<0.3 or abs(slope)>10: continue
        if slope<0 and max(x1,x2)<w*0.55:
            left.append((x1,y1,x2,y2))
        elif slope>0 and min(x1,x2)>w*0.45:
            right.append((x1,y1,x2,y2))
    return left,right

# ---------- RANSAC 기반 곡선 ----------
def fit_lane_curve_ransac(segments, degree=2):
    if len(segments)==0: return None
    xs, ys = [],[]
    for x1,y1,x2,y2 in segments:
        xs += [x1,x2]; ys += [y1,y2]
    xs = np.array(xs); ys = np.array(ys)
    if len(xs)<degree+1: return None
    ys_reshaped = ys.reshape(-1,1)
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(LinearRegression()))
    try: model.fit(ys_reshaped,xs)
    except: return None
    return model

def get_lane_points(model, y_bottom, y_top):
    length = abs(y_bottom-y_top)+1
    y_vals = np.linspace(y_bottom,y_top,length)
    x_vals = model.predict(y_vals.reshape(-1,1)).astype(int)
    return x_vals, y_vals

# ---------- 곡선 오버레이 ----------
def draw_lane_overlay_curve(frame, left_segments, right_segments, prev_left_curve, prev_right_curve, Minv):
    h = frame.shape[0]
    y_bottom = int(h*ROI_BOTTOM_Y)
    y_top = int(h*ROI_TOP_Y)

    left_model = fit_lane_curve_ransac(left_segments) or prev_left_curve
    right_model = fit_lane_curve_ransac(right_segments) or prev_right_curve

    if left_model is None or right_model is None:
        return frame, prev_left_curve, prev_right_curve

    left_x, y_vals = get_lane_points(left_model, y_bottom, y_top)
    right_x, _     = get_lane_points(right_model, y_bottom, y_top)

    pts_left = np.vstack([left_x, y_vals]).T
    pts_right = np.vstack([right_x, y_vals]).T[::-1]
    pts = np.vstack([pts_left, pts_right])
    pts = np.array([pts],dtype=np.int32)

    overlay = np.zeros_like(frame)
    cv2.fillPoly(overlay,pts,(180,255,255))
    # 역투시
    overlay_warped = cv2.warpPerspective(overlay,Minv,(frame.shape[1],frame.shape[0]))
    result = cv2.addWeighted(frame,1.0,overlay_warped,0.4,0)

    # 미리보기 박스
    lane_preview = cv2.resize(threshold_binary(frame),(frame.shape[1]//3,frame.shape[0]//3))
    result[10:10+lane_preview.shape[0],frame.shape[1]-10-lane_preview.shape[1]:frame.shape[1]-10] = cv2.cvtColor(lane_preview,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result,(frame.shape[1]-10-lane_preview.shape[1],10),(frame.shape[1]-10,10+lane_preview.shape[0]),(0,0,0),2)

    return result,left_model,right_model

# ---------- 메인 ----------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:",VIDEO_PATH)
        return

    prev_left_curve, prev_right_curve = None,None

    while True:
        ret, frame = cap.read()
        if not ret: break

        # ROI 투시 변환
        warped_frame, M, Minv = warp_roi(frame)

        # 차선 검출
        lane_mask = threshold_binary(warped_frame)
        lines = detect_lines(lane_mask)
        left,right = separate_left_right(lines, warped_frame.shape)

        vis, prev_left_curve, prev_right_curve = draw_lane_overlay_curve(
            frame,left,right,prev_left_curve,prev_right_curve,Minv
        )

        cv2.imshow("Lane Detection Smooth Curve",cv2.resize(vis,(int(vis.shape[1]*SCALE),int(vis.shape[0]*SCALE))))
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
