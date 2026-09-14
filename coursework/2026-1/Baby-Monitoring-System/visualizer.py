# visualizer.py
import cv2
import numpy as np
from korean_text import put_korean_text

# ── 배경 제거기 ────────────────────────────────────────
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50,
    detectShadows=True)

prev_gray_feat = None


# ── 라벨 추가 ──────────────────────────────────────────
def _add_label(frame, title, subtitle, color):
    frame = put_korean_text(
        frame, title, (10, 8),
        font_size  = 18,
        color      = color,
        background = (0, 0, 0)
    )
    frame = put_korean_text(
        frame, subtitle, (10, 32),
        font_size  = 15,
        color      = color,
        background = (0, 0, 0)
    )
    return frame


# ── Edge 추출 ──────────────────────────────────────────
def extract_edge(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    result  = frame.copy()
    result[edges > 0] = [0, 255, 0]

    result = _add_label(result,
                        "② Edge Detection (Canny)",
                        "경계선 추출",
                        (0, 255, 0))

    edge_count = np.sum(edges > 0)
    result = put_korean_text(
        result,
        f"Edge 픽셀 수: {edge_count:,}",
        (10, result.shape[0]-40),
        font_size  = 15,
        color      = (0, 255, 0),
        background = (0, 0, 0)
    )
    return result, edges


# ── Boundary 추출 ──────────────────────────────────────
def extract_boundary(frame, bed_zones=None):
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    result  = frame.copy()
    h, w    = gray.shape

    if bed_zones and len(bed_zones) > 0:
        mask   = np.zeros((h,w), dtype=np.uint8)
        for pts in bed_zones:
            cv2.fillPoly(mask, [pts], 255)
        masked = cv2.bitwise_and(blurred, blurred,
                                  mask=mask)
    else:
        masked = blurred

    _, thresh   = cv2.threshold(
        masked, 0, 255,
        cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    small, medium, large = [], [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if   area < 200:  small.append(cnt)
        elif area < 2000: medium.append(cnt)
        else:             large.append(cnt)

    cv2.drawContours(result, small,  -1,
                     (100,100,100), 1)
    cv2.drawContours(result, medium, -1,
                     (0,200,255), 1)
    cv2.drawContours(result, large,  -1,
                     (0,0,255), 2)

    for cnt in large:
        x,y,bw,bh = cv2.boundingRect(cnt)
        area       = cv2.contourArea(cnt)
        cv2.rectangle(result, (x,y),
                      (x+bw,y+bh), (0,0,255), 1)
        result = put_korean_text(
            result, f"{area:.0f}px",
            (x, y-5),
            font_size = 14,
            color     = (0, 0, 255)
        )

    result = _add_label(result,
                        "③ Boundary (Contour)",
                        "윤곽선 추출",
                        (0, 200, 255))
    result = put_korean_text(
        result,
        f"대:{len(large)} 중:{len(medium)} 소:{len(small)}",
        (10, result.shape[0]-40),
        font_size  = 15,
        color      = (0, 200, 255),
        background = (0, 0, 0)
    )
    return result, large


# ── Segmentation ───────────────────────────────────────
def segment_frame(frame, bed_zones=None):
    fg_mask = bg_subtractor.apply(frame)
    result  = frame.copy()
    h, w    = frame.shape[:2]

    if bed_zones and len(bed_zones) > 0:
        bed_mask = np.zeros((h,w), dtype=np.uint8)
        for pts in bed_zones:
            cv2.fillPoly(bed_mask, [pts], 255)
        fg_mask = cv2.bitwise_and(
            fg_mask, fg_mask, mask=bed_mask)

    kernel  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5,5))
    fg_mask = cv2.morphologyEx(
        fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(
        fg_mask, cv2.MORPH_CLOSE, kernel)

    overlay = result.copy()
    bg_mask = cv2.bitwise_not(fg_mask)
    overlay[bg_mask > 0] = [150, 100, 50]
    overlay[fg_mask > 0] = [0,   0,   200]
    cv2.addWeighted(overlay, 0.4,
                    result,  0.6, 0, result)

    contours, _ = cv2.findContours(
        fg_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    baby_found = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            baby_found = True
            cv2.drawContours(result, [cnt], -1,
                             (0,0,255), 2)

    result = _add_label(result,
                        "④ Segmentation (MOG2)",
                        "배경/전경 분리",
                        (0, 0, 255))

    fg_ratio = np.sum(fg_mask > 0) / (h*w) * 100
    baby_txt = "감지됨" if baby_found else "없음"
    result   = put_korean_text(
        result,
        f"전경: {fg_ratio:.1f}%  아기: {baby_txt}",
        (10, result.shape[0]-40),
        font_size  = 15,
        color      = (0, 0, 255),
        background = (0, 0, 0)
    )
    return result, fg_mask


# ── Feature Detection ──────────────────────────────────
prev_gray_feat = None

def detect_features(frame, bed_zones=None):
    global prev_gray_feat

    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = frame.copy()
    h, w   = gray.shape

    # ── 마스크 생성 ─────────────────────────────────
    if bed_zones and len(bed_zones) > 0:
        mask = np.zeros((h, w), dtype=np.uint8)  # ← dtype 명시
        for pts in bed_zones:
            # 포인트가 frame 범위 안에 있는지 클리핑
            clipped = np.clip(pts, [0, 0], [w-1, h-1])
            cv2.fillPoly(mask, [clipped], 255)

        # 마스크가 완전히 비어있으면 None 처리
        if cv2.countNonZero(mask) == 0:
            mask     = None
            gray_roi = gray
        else:
            gray_roi = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        mask     = None
        gray_roi = gray

    # ── Harris Corner ────────────────────────────────
    try:
        gray_f  = np.float32(gray_roi)
        corners = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        result[corners > 0.01 * corners.max()] = [0, 0, 255]
    except Exception as e:
        print(f"[Harris] 오류: {e}")
        corners = None

    # ── Optical Flow ─────────────────────────────────
    if prev_gray_feat is not None:

        # ✅ 핵심 수정: 해상도 불일치 시 prev 초기화
        if prev_gray_feat.shape != gray.shape:
            print("[OptFlow] 해상도 불일치 → prev 초기화")
            prev_gray_feat = gray.copy()
            return result

        try:
            pts = cv2.goodFeaturesToTrack(
                prev_gray_feat,
                maxCorners   = 50,
                qualityLevel = 0.3,
                minDistance  = 10,
                mask         = mask   # None 또는 uint8 마스크
            )

            if pts is not None and len(pts) > 0:
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray_feat, gray, pts, None
                )

                if new_pts is not None and status is not None:
                    good_new = new_pts[status == 1]
                    good_old = pts[status == 1]

                    for new, old in zip(good_new, good_old):
                        nx, ny = new.ravel().astype(int)
                        ox, oy = old.ravel().astype(int)

                        # 범위 체크
                        if not (0 <= nx < w and 0 <= ny < h):
                            continue
                        if not (0 <= ox < w and 0 <= oy < h):
                            continue

                        dist = np.sqrt((nx-ox)**2 + (ny-oy)**2)
                        if dist > 1:
                            intensity = min(int(dist * 10), 255)
                            color     = (0, 255-intensity, intensity)
                            cv2.arrowedLine(
                                result,
                                (ox, oy), (nx, ny),
                                color, 1, tipLength=0.4
                            )
                            cv2.circle(result, (nx, ny),
                                       3, (0, 255, 0), -1)

        except cv2.error as e:
            print(f"[OptFlow] OpenCV 오류: {e}")
        except Exception as e:
            print(f"[OptFlow] 오류: {e}")

    # prev 업데이트
    prev_gray_feat = gray.copy()

    # ── 라벨 표시 ────────────────────────────────────
    from korean_text import put_korean_text
    result = _add_label(result,
                        "⑤ Feature (Harris+OptFlow)",
                        "특징점 + 움직임 벡터",
                        (255, 100, 0))
    if corners is not None:
        corner_count = int(np.sum(corners > 0.01 * corners.max()))
        result = put_korean_text(
            result,
            f"코너 수: {corner_count:,}",
            (10, result.shape[0]-40),
            font_size  = 15,
            color      = (255, 100, 0),
            background = (0, 0, 0)
        )
    return result


# ── 대시보드 ───────────────────────────────────────────
def make_dashboard(original, edge_result,
                   boundary_result, segment_result,
                   feature_result, integrated):
    ph, pw = 360, 480

    panels = [
        (original,        "① 원본",            (200,200,200)),
        (edge_result,     "② Edge 경계선",      (0,  255,  0)),
        (boundary_result, "③ Boundary 윤곽선",  (0,  200,255)),
        (segment_result,  "④ Segmentation",    (0,  0,  255)),
        (feature_result,  "⑤ Feature 특징점",  (255,100,   0)),
        (integrated,      "⑥ 통합 결과",        (255,255,   0)),
    ]

    imgs = []
    for img, label, color in panels:
        if img is None:
            panel = np.zeros((ph,pw,3), dtype=np.uint8)
        else:
            panel = cv2.resize(img, (pw, ph))

        cv2.rectangle(panel, (0,0),
                      (pw-1,ph-1), color, 2)
        cv2.rectangle(panel, (0,0), (pw,30),
                      (0,0,0), -1)
        panel = put_korean_text(
            panel, label, (8, 5),
            font_size = 18,
            color     = color
        )
        imgs.append(panel)

    top   = np.hstack(imgs[:3])
    bot   = np.hstack(imgs[3:])
    board = np.vstack([top, bot])

    title_bar = np.zeros(
        (45, board.shape[1], 3), dtype=np.uint8)
    title_bar = put_korean_text(
        title_bar,
        "Baby Monitor — 알고리즘 대시보드  "
        "(V키: 닫기)",
        (15, 8),
        font_size = 22,
        color     = (255, 255, 0)
    )
    return np.vstack([title_bar, board])