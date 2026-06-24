# privacy.py
import cv2
import numpy as np

# ── 전역 변수 ──────────────────────────────────────────
# 현재 찍고 있는 점들
current_points  = []

# 확정된 다각형 영역들 [(points_array, label), ...]
confirmed_faces = []

# 미리보기용 현재 마우스 위치
mouse_pos       = (0, 0)

# 포인트 최소 개수 (삼각형 이상)
MIN_POINTS      = 3


# ── 마우스 콜백 ────────────────────────────────────────
def mouse_callback_privacy(event, x, y, flags, param):
    global current_points, confirmed_faces, mouse_pos

    # 마우스 이동 → 현재 위치 업데이트 (미리보기용)
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_pos = (x, y)

    # 좌클릭 → 점 추가
    elif event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        print(f"📍 점 추가: ({x}, {y}) | "
              f"현재 {len(current_points)}개 "
              f"(Enter로 확정, 최소 {MIN_POINTS}개 필요)")

    # 우클릭 → 마지막 점 삭제
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            removed = current_points.pop()
            print(f"🗑 점 삭제: {removed} | "
                  f"남은 점: {len(current_points)}개")
        else:
            print("❌ 삭제할 점이 없습니다")


# ── 키보드 입력 처리 ───────────────────────────────────
def handle_privacy_key(key):
    """
    Enter : 현재 점들로 영역 확정
    R     : 현재 작업 중인 점 초기화
    D     : 마지막으로 확정된 영역 삭제
    C     : 모든 영역 초기화
    """
    global current_points, confirmed_faces

    # Enter 키 → 영역 확정
    if key == 13:
        if len(current_points) >= MIN_POINTS:
            pts = np.array(current_points, dtype=np.int32)
            confirmed_faces.append(pts)
            print(f"✅ 영역 확정! | "
                  f"총 {len(confirmed_faces)}개 영역 등록됨")
            current_points = []  # 현재 작업 초기화
        else:
            print(f"⚠️  점이 부족합니다 | "
                  f"현재 {len(current_points)}개 "
                  f"(최소 {MIN_POINTS}개 필요)")

    # R 키 → 현재 작업 중인 점만 초기화
    elif key == ord('r'):
        current_points = []
        print("🔄 현재 작업 중인 점 초기화")

    # D 키 → 마지막 확정 영역 삭제
    elif key == ord('d'):
        if confirmed_faces:
            confirmed_faces.pop()
            print(f"🗑 마지막 영역 삭제 | "
                  f"남은 영역: {len(confirmed_faces)}개")
        else:
            print("❌ 삭제할 영역이 없습니다")

    # C 키 → 전체 초기화
    elif key == ord('c'):
        current_points  = []
        confirmed_faces = []
        print("🔄 모든 영역 초기화")


# ── 블러 처리 ──────────────────────────────────────────
def apply_polygon_blur(frame, points, blur_mode="mosaic"):
    """다각형 영역에 블러 적용"""

    result = frame.copy()
    h, w   = frame.shape[:2]

    # 다각형 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    if blur_mode == "mosaic":
        pixel_size  = 20
        temp        = frame.copy()
        # 픽셀화 처리
        for y in range(0, h, pixel_size):
            for x in range(0, w, pixel_size):
                y2  = min(y + pixel_size, h)
                x2  = min(x + pixel_size, w)
                roi = frame[y:y2, x:x2]
                avg = roi.mean(axis=(0,1)).astype(np.uint8)
                temp[y:y2, x:x2] = avg
        # 마스크 적용
        result = np.where(
            mask[:,:,np.newaxis] == 255,
            temp, result
        )

    elif blur_mode == "blur":
        blurred = cv2.GaussianBlur(frame, (99, 99), 30)
        result  = np.where(
            mask[:,:,np.newaxis] == 255,
            blurred, result
        )

    elif blur_mode == "black":
        result[mask == 255] = 0

    elif blur_mode == "emoji":
        # 다각형 중심에 이모지 표시
        M  = cv2.moments(points)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = points.mean(axis=0).astype(int)
        result[mask == 255] = [50, 50, 50]
        cv2.putText(result, ":)", (cx-15, cy+10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 200, 255), 3)

    return result


# ── 작업 중인 점 미리보기 ──────────────────────────────
def draw_preview(frame):
    """
    현재 찍고 있는 점들과
    마우스 위치까지의 선을 미리보기로 표시
    """
    result = frame.copy()

    if not current_points:
        return result

    pts = current_points

    # 점 그리기
    for i, pt in enumerate(pts):
        # 점 원
        cv2.circle(result, pt, 6, (0, 200, 255), -1)
        cv2.circle(result, pt, 6, (255, 255, 255), 1)
        # 점 번호
        cv2.putText(result, str(i+1),
                    (pt[0]+8, pt[1]-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 1)

    # 점들 사이 선 연결
    for i in range(len(pts) - 1):
        cv2.line(result, pts[i], pts[i+1],
                 (0, 200, 255), 2)

    # 마지막 점 → 현재 마우스 위치 점선 미리보기
    if mouse_pos:
        # 점선 효과
        pt1 = pts[-1]
        pt2 = mouse_pos
        draw_dashed_line(result, pt1, pt2, (0,200,255))

        # 점 3개 이상이면 닫히는 선도 미리보기
        if len(pts) >= MIN_POINTS:
            draw_dashed_line(result, mouse_pos, pts[0],
                             (0, 255, 0))
            cv2.putText(result,
                        "Enter: 영역 확정",
                        (mouse_pos[0]+10, mouse_pos[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

    # 안내 텍스트
    guide = (f"점 {len(pts)}개 | "
             f"좌클릭: 점 추가 | "
             f"우클릭: 점 삭제 | "
             f"Enter: 확정")
    cv2.putText(result, guide, (10, frame.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 255), 1)

    return result


def draw_dashed_line(frame, pt1, pt2,
                     color, gap=10):
    """점선 그리기"""
    dist  = np.linalg.norm(np.array(pt2) - np.array(pt1))
    if dist == 0:
        return
    steps = int(dist / gap)
    for i in range(steps):
        if i % 2 == 0:
            s = np.array(pt1) + (np.array(pt2)-np.array(pt1)) * (i/steps)
            e = np.array(pt1) + (np.array(pt2)-np.array(pt1)) * ((i+1)/steps)
            cv2.line(frame,
                     tuple(s.astype(int)),
                     tuple(e.astype(int)),
                     color, 1)


# ── 메인 사생활 보호 함수 ──────────────────────────────
def apply_privacy(frame, privacy_on, blur_mode="mosaic"):

    result = frame.copy()

    # 사생활 보호 OFF
    if not privacy_on:
        cv2.putText(result, "Privacy: OFF",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
        return result, 0

    # 확정된 영역 블러 처리
    for pts in confirmed_faces:
        result = apply_polygon_blur(result, pts, blur_mode)
        # 다각형 테두리 표시
        cv2.polylines(result, [pts], True,
                      (0, 165, 255), 2)

    # 현재 작업 중인 점 미리보기
    result = draw_preview(result)

    # 상태 표시
    count = len(confirmed_faces)
    cv2.putText(result,
                f"Privacy: ON | {count}개 영역 보호중",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 165, 255), 2)

    return result, count