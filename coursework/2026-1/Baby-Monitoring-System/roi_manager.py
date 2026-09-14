# roi_manager.py
import cv2
import numpy as np
from korean_text import put_korean_text

# ── 모드 상수 ──────────────────────────────────────────
MODE_NONE = "none"
MODE_FACE = "face"
MODE_BED  = "bed"

# ── 전역 상태 ──────────────────────────────────────────
current_mode   = MODE_NONE
current_points = []
mouse_pos      = (0, 0)
face_zones     = []
bed_zones      = []
MIN_POINTS     = 3

COLOR = {
    MODE_FACE : (0,  165, 255),
    MODE_BED  : (255, 200,   0),
    MODE_NONE : (200, 200, 200),
}

LABEL = {
    MODE_FACE : "👤 얼굴 구역 지정중",
    MODE_BED  : "🛏 침대 구역 지정중",
}


# ── 마우스 콜백 ────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global current_points, mouse_pos

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if current_mode == MODE_NONE:
            print("⚠️  F(얼굴) 또는 B(침대) 키로 모드를 먼저 선택하세요!")
            return
        current_points.append((x, y))
        print(f"📍 [{LABEL[current_mode]}] "
              f"점 추가: ({x},{y}) | "
              f"총 {len(current_points)}개")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            removed = current_points.pop()
            print(f"🗑 점 삭제: {removed} | "
                  f"남은: {len(current_points)}개")


# ── 키보드 처리 ────────────────────────────────────────
def handle_key(key):
    global current_mode, current_points
    global face_zones, bed_zones

    if key == ord('f'):
        current_mode   = MODE_FACE
        current_points = []
        print("\n" + "="*45)
        print("  👤 얼굴 구역 지정 모드")
        print("  좌클릭으로 얼굴 주변에 점을 찍으세요")
        print("  Enter: 확정 | 우클릭: 점삭제 | R: 초기화")
        print("="*45)

    elif key == ord('b'):
        current_mode   = MODE_BED
        current_points = []
        print("\n" + "="*45)
        print("  🛏 침대 구역 지정 모드")
        print("  좌클릭으로 침대 모서리에 점을 찍으세요")
        print("  Enter: 확정 | 우클릭: 점삭제 | R: 초기화")
        print("="*45)

    elif key == 13:  # Enter
        if len(current_points) < MIN_POINTS:
            print(f"⚠️  점 부족! "
                  f"({len(current_points)}개 / "
                  f"최소 {MIN_POINTS}개)")
            return
        pts = np.array(current_points, dtype=np.int32)
        if current_mode == MODE_FACE:
            face_zones.append(pts)
            print(f"✅ 얼굴 구역 확정! "
                  f"(총 {len(face_zones)}개)")
        elif current_mode == MODE_BED:
            bed_zones.append(pts)
            print(f"✅ 침대 구역 확정! "
                  f"(총 {len(bed_zones)}개)")
        current_points = []

    elif key == ord('r'):
        current_points = []
        print("🔄 현재 작업 점 초기화")

    elif key == ord('d'):
        if current_mode == MODE_FACE and face_zones:
            face_zones.pop()
            print(f"🗑 얼굴 구역 삭제 | "
                  f"남은: {len(face_zones)}개")
        elif current_mode == MODE_BED and bed_zones:
            bed_zones.pop()
            print(f"🗑 침대 구역 삭제 | "
                  f"남은: {len(bed_zones)}개")
        else:
            print("❌ 삭제할 구역 없음")

    elif key == ord('c'):
        current_points = []
        face_zones     = []
        bed_zones      = []
        current_mode   = MODE_NONE
        print("🔄 모든 구역 초기화")

    elif key == 27:  # ESC
        current_mode   = MODE_NONE
        current_points = []
        print("⏹ 모드 해제")


# ── 점선 그리기 ────────────────────────────────────────
def draw_dashed_line(frame, pt1, pt2, color, gap=10):
    dist = np.linalg.norm(
        np.array(pt2) - np.array(pt1))
    if dist == 0:
        return
    steps = int(dist / gap)
    for i in range(steps):
        if i % 2 == 0:
            s = np.array(pt1) + \
                (np.array(pt2)-np.array(pt1))*(i/steps)
            e = np.array(pt1) + \
                (np.array(pt2)-np.array(pt1))*((i+1)/steps)
            cv2.line(frame,
                     tuple(s.astype(int)),
                     tuple(e.astype(int)),
                     color, 1)


# ── 점 찍기 미리보기 ───────────────────────────────────
def draw_preview(frame):
    result = frame.copy()
    if not current_points:
        return result

    color = COLOR.get(current_mode, COLOR[MODE_NONE])
    pts   = current_points

    for i, pt in enumerate(pts):
        cv2.circle(result, pt, 7, color, -1)
        cv2.circle(result, pt, 7, (255,255,255), 1)
        result = put_korean_text(
            result, str(i+1),
            (pt[0]+9, pt[1]-20),
            font_size = 16,
            color     = color
        )

    for i in range(len(pts)-1):
        cv2.line(result, pts[i], pts[i+1], color, 2)

    if mouse_pos:
        draw_dashed_line(result, pts[-1],
                         mouse_pos, color)
        if len(pts) >= MIN_POINTS:
            draw_dashed_line(result, mouse_pos,
                             pts[0], (0,255,0))
            result = put_korean_text(
                result, "Enter: 구역 확정",
                (mouse_pos[0]+10, mouse_pos[1]-10),
                font_size = 16,
                color     = (0, 255, 0)
            )

    guide = (f"점 {len(pts)}개 | "
             f"좌클릭: 점추가 | "
             f"우클릭: 점삭제 | "
             f"Enter: 확정")
    result = put_korean_text(
        result, guide,
        (10, result.shape[0]-35),
        font_size  = 16,
        color      = color,
        background = (0, 0, 0)
    )
    return result


# ── 확정된 구역 표시 ───────────────────────────────────
def draw_zones(frame):
    result = frame.copy()

    for i, pts in enumerate(face_zones):
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], (0, 100, 200))
        cv2.addWeighted(overlay, 0.15,
                        result,  0.85, 0, result)
        cv2.polylines(result, [pts], True,
                      COLOR[MODE_FACE], 2)
        cx = int(pts[:,0].mean())
        cy = int(pts[:,1].mean())
        result = put_korean_text(
            result, f"얼굴{i+1}",
            (cx-25, cy-10),
            font_size = 18,
            color     = COLOR[MODE_FACE]
        )

    for i, pts in enumerate(bed_zones):
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], (200, 150, 0))
        cv2.addWeighted(overlay, 0.15,
                        result,  0.85, 0, result)
        cv2.polylines(result, [pts], True,
                      COLOR[MODE_BED], 2)
        cx = int(pts[:,0].mean())
        cy = int(pts[:,1].mean())
        result = put_korean_text(
            result, f"침대{i+1}",
            (cx-25, cy-10),
            font_size = 18,
            color     = COLOR[MODE_BED]
        )

    return result


# ── 얼굴 블러 처리 ─────────────────────────────────────
def apply_face_privacy(frame, privacy_on,
                        blur_mode="mosaic"):
    result = frame.copy()
    if not privacy_on:
        return result

    for pts in face_zones:
        h, w  = frame.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        if blur_mode == "mosaic":
            pixel_size = 20
            temp       = frame.copy()
            for y in range(0, h, pixel_size):
                for x in range(0, w, pixel_size):
                    y2  = min(y+pixel_size, h)
                    x2  = min(x+pixel_size, w)
                    roi = frame[y:y2, x:x2]
                    avg = roi.mean(axis=(0,1)
                                   ).astype(np.uint8)
                    temp[y:y2, x:x2] = avg
            result = np.where(
                mask[:,:,np.newaxis]==255,
                temp, result)

        elif blur_mode == "blur":
            blurred = cv2.GaussianBlur(
                frame, (99,99), 30)
            result  = np.where(
                mask[:,:,np.newaxis]==255,
                blurred, result)

        elif blur_mode == "black":
            result[mask==255] = 0

        elif blur_mode == "emoji":
            M = cv2.moments(pts)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                cx,cy = pts.mean(axis=0).astype(int)
            result[mask==255] = [40, 40, 40]
            cv2.putText(result, ":)",
                        (cx-15, cy+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,200,255), 3)

        cv2.polylines(result, [pts], True,
                      COLOR[MODE_FACE], 2)

    return result


# ── 침대 구역 움직임 감지 ──────────────────────────────
prev_frame_bed = None

def detect_movement_in_bed(gray_frame, frame,
                            threshold=3000):
    global prev_frame_bed

    result    = frame.copy()
    h, w      = gray_frame.shape[:2]
    is_moving = False
    score     = 0

    if prev_frame_bed is None:
        prev_frame_bed = gray_frame
        return score, is_moving, result

    diff      = cv2.absdiff(prev_frame_bed, gray_frame)
    _, thresh = cv2.threshold(
        diff, 25, 255, cv2.THRESH_BINARY)

    if len(bed_zones) == 0:
        score     = int(np.sum(thresh)/255)
        is_moving = score > threshold
    else:
        bed_mask  = np.zeros((h,w), dtype=np.uint8)
        for pts in bed_zones:
            cv2.fillPoly(bed_mask, [pts], 255)
        masked    = cv2.bitwise_and(
            thresh, thresh, mask=bed_mask)
        score     = int(np.sum(masked)/255)
        is_moving = score > threshold

        if is_moving:
            contours, _ = cv2.findContours(
                masked,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    x,y,bw,bh = cv2.boundingRect(cnt)
                    cv2.rectangle(result,
                                  (x,y),(x+bw,y+bh),
                                  (0,0,255), 1)

    prev_frame_bed = gray_frame
    return score, is_moving, result