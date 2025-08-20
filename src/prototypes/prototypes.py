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

# ---------- ROI ----------
def region_of_interest(img):
    h, w = img.shape[:2]

    # 위쪽 좌우폭 기존 대비 조금 넓혀 투시감 강화
    top_left_x = int(w*0.35)  # 기존 0.35에서 그대로
    top_right_x = int(w*0.65) # 기존 0.65에서 그대로

    # 아래쪽 좌우폭 그대로
    bottom_left_x = int(w*0.1)
    bottom_right_x = int(w*0.9)

    pts = np.array([[
        (bottom_left_x, int(h*ROI_BOTTOM_Y)),  # 아래 왼쪽
        (top_left_x, int(h*ROI_TOP_Y)),        # 위 왼쪽
        (top_right_x, int(h*ROI_TOP_Y)),       # 위 오른쪽
        (bottom_right_x, int(h*ROI_BOTTOM_Y))  # 아래 오른쪽
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255 if img.ndim==2 else (255,255,255))
    return cv2.bitwise_and(img, mask), pts


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
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=30,maxLineGap=60)
    segments=[]
    if lines is None: return segments, edges
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if x2==x1: continue
        slope = (y2-y1)/(x2-x1+1e-6)
        if abs(slope)<0.3 or abs(slope)>10: continue
        segments.append((x1,y1,x2,y2))
    return segments, edges

# ---------- 점선 필터 ----------
def filter_dashed_pattern(segments, max_gap=20):
    if not segments: return []
    segments = sorted(segments,key=lambda b:min(b[1],b[3]))
    filtered=[]
    prev_bottom=None
    for x1,y1,x2,y2 in segments:
        top=min(y1,y2)
        bottom=max(y1,y2)
        if prev_bottom is None or top-prev_bottom<=max_gap:
            filtered.append((x1,y1,x2,y2))
            prev_bottom=bottom
        else:
            prev_bottom=bottom
    if len(filtered)<2: return []
    return filtered

# ---------- 박스 연결 ----------
def merge_and_filter_boxes(boxes):
    if not boxes: return []
    boxes = sorted(boxes,key=lambda b:b[1])
    merged=[]
    current=boxes[0]
    for b in boxes[1:]:
        extended_current=[current[0], current[1]-5, current[2], current[3]+5]
        if not (b[0] > extended_current[2] or b[2] < extended_current[0] or b[1] > extended_current[3] or b[3] < extended_current[1]):
            current=[min(current[0],b[0]),min(current[1],b[1]),
                     max(current[2],b[2]),max(current[3],b[3])]
            if current[3]-current[1] > 3*(b[3]-b[1]):
                current=b
        else:
            merged.append(current)
            current=b
    merged.append(current)
    return merged

# ---------- 버드아이뷰 박스 표시 ----------
def draw_lane_boxes_birdeye(warped, segments):
    boxes=[]
    for x1,y1,x2,y2 in segments:
        top=min(y1,y2); bottom=max(y1,y2)
        left=min(x1,x2)-5; right=max(x1,x2)+5
        boxes.append([left,top,right,bottom])
    merged_boxes = merge_and_filter_boxes(boxes)
    overlay = np.zeros_like(warped)
    for left,top,right,bottom in merged_boxes:
        cv2.rectangle(overlay,(left,top),(right,bottom),(0,255,255),-1) # 하늘색 반투명
    result = cv2.addWeighted(warped,1.0,overlay,0.4,0)
    return result, overlay

# ---------- 메인 ----------
def main():
    cap=cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다:", VIDEO_PATH)
        return

    while True:
        ret, frame=cap.read()
        if not ret: break

        roi_frame, roi_pts = region_of_interest(frame)
        warped_roi, M, Minv = get_perspective_transform(frame, roi_pts[0])
        segments, edges = detect_lanes_birdeye(warped_roi)
        segments = filter_dashed_pattern(segments)

        # --- 버드아이뷰 박스 ---
        vis_birdeye, birdeye_overlay = draw_lane_boxes_birdeye(warped_roi, segments)

        # --- 원본 화면 위에 투명 오버레이 ---
        overlay_back = cv2.warpPerspective(birdeye_overlay, Minv, (frame.shape[1], frame.shape[0]))
        combined = cv2.addWeighted(frame,0.7,overlay_back,0.3,0)

        # --- 오른쪽 위 작은 ROI (원본 투영) ---
        roi_h, roi_w = vis_birdeye.shape[:2]
        small_scale=0.3
        x_offset=frame.shape[1]-int(roi_w*small_scale)-10
        y_offset=10
        small_roi = cv2.resize(vis_birdeye,(int(roi_w*small_scale), int(roi_h*small_scale)))
        combined[y_offset:y_offset+small_roi.shape[0], x_offset:x_offset+small_roi.shape[1]] = small_roi
        cv2.rectangle(combined,(x_offset,y_offset),(x_offset+small_roi.shape[1],y_offset+small_roi.shape[0]),(0,0,255),2)
        cv2.putText(combined,"Warped ROI",(x_offset+5,y_offset+25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        # --- 그 아래 Canny 엣지 오버레이 ---
        edges_color = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        edges_small = cv2.resize(edges_color,(int(roi_w*small_scale), int(roi_h*small_scale)))
        y_offset += small_roi.shape[0]+5
        combined[y_offset:y_offset+edges_small.shape[0], x_offset:x_offset+edges_small.shape[1]] = edges_small
        cv2.rectangle(combined,(x_offset,y_offset),(x_offset+edges_small.shape[1],y_offset+edges_small.shape[0]),(0,255,0),2)
        cv2.putText(combined,"Canny ROI",(x_offset+5,y_offset+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # --- 그 아래 버드아이뷰 박스 오버레이 ---
        birdeye_small = cv2.resize(vis_birdeye,(int(roi_w*small_scale), int(roi_h*small_scale)))
        y_offset += edges_small.shape[0]+5
        combined[y_offset:y_offset+birdeye_small.shape[0], x_offset:x_offset+birdeye_small.shape[1]] = birdeye_small
        cv2.rectangle(combined,(x_offset,y_offset),(x_offset+birdeye_small.shape[1],y_offset+birdeye_small.shape[0]),(255,255,0),2)
        cv2.putText(combined,"Lane Boxes",(x_offset+5,y_offset+20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.imshow("Lane Detection Full Overlay", cv2.resize(combined,(int(frame.shape[1]*SCALE), int(frame.shape[0]*SCALE))))
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
