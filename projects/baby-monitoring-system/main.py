# main.py
import cv2
import numpy as np
import datetime
from step2_preprocess import preprocess
from korean_text      import put_korean_text, put_status_bar
from roi_manager      import (
    mouse_callback, handle_key,
    draw_zones, draw_preview,
    apply_face_privacy,
    detect_movement_in_bed,
    current_mode, face_zones, bed_zones,
    COLOR, LABEL, MODE_NONE
)
from visualizer import (
    extract_edge, extract_boundary,
    segment_frame, detect_features,
    make_dashboard
)

# ── 설정 ───────────────────────────────────────────────
VIDEO_PATH  = "../../../data/videos/baby_test2.mp4"
VIDEO_MODE     = "night"   # "night" or "day"
MOVE_THRESHOLD = 3000      # 움직임 감지 민감도

# ── 상태 변수 ──────────────────────────────────────────
privacy_on     = True
blur_mode      = "mosaic"
BLUR_MODES     = ["mosaic", "blur", "black", "emoji"]
blur_index     = 0
paused         = False
show_dashboard = False


def print_guide():
    print("\n" + "="*50)
    print("  🍼 Baby Monitor — 키보드 단축키")
    print("="*50)
    print("  Q      : 종료")
    print("  SPACE  : 일시정지 / 재생")
    print("  S      : 스냅샷 저장")
    print("  V      : 알고리즘 대시보드 ON/OFF")
    print("  ─────────────────────────────────────")
    print("  P      : 사생활 보호 ON/OFF")
    print("  M      : 블러 방식 변경")
    print("         (mosaic→blur→black→emoji)")
    print("  ─────────────────────────────────────")
    print("  F      : 얼굴 구역 지정 모드")
    print("  B      : 침대 구역 지정 모드")
    print("  좌클릭  : 점 추가")
    print("  우클릭  : 마지막 점 삭제")
    print("  Enter  : 구역 확정")
    print("  R      : 현재 작업 점 초기화")
    print("  D      : 마지막 확정 구역 삭제")
    print("  C      : 전체 구역 초기화")
    print("  ESC    : 모드 해제")
    print("="*50 + "\n")


print_guide()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
    exit()

fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"✅ 영상 로드 성공!")
print(f"   해상도: {width}x{height} | "
      f"FPS: {fps} | "
      f"총 {total}프레임 ({total/fps:.1f}초)\n")

cv2.namedWindow("Baby Monitor")
cv2.setMouseCallback("Baby Monitor", mouse_callback)

try:
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # ── 전처리 ─────────────────────────────────
            gray, processed = preprocess(frame, VIDEO_MODE)

            # ── 알고리즘 실행 ───────────────────────────
            edge_result,     _  = extract_edge(frame)
            boundary_result, _  = extract_boundary(
                                       frame, bed_zones)
            segment_result,  _  = segment_frame(
                                       frame, bed_zones)
            feature_result      = detect_features(
                                       frame, bed_zones)

            # ── 통합 결과 ───────────────────────────────
            integrated = draw_zones(frame.copy())
            integrated = apply_face_privacy(
                             integrated,
                             privacy_on, blur_mode)
            score, moving, integrated = \
                detect_movement_in_bed(
                    processed, integrated,
                    threshold=MOVE_THRESHOLD)
            integrated = draw_preview(integrated)

            # ── 상태바 (한글) ───────────────────────────
            now        = datetime.datetime.now().strftime(
                         "%Y-%m-%d %H:%M:%S")
            mode_color = COLOR.get(current_mode,
                                    (200,200,200))
            mode_label = LABEL.get(current_mode,
                                    "없음 (F/B키로 선택)")
            p_color    = (0,165,255) if privacy_on \
                         else (0,255,0)
            p_text     = "ON" if privacy_on else "OFF"
            m_color    = (0,0,255) if moving \
                         else (0,255,0)
            m_text     = "🔴 움직임 감지!" if moving \
                         else "🟢 안정"
            face_cnt   = len(face_zones)
            bed_cnt    = len(bed_zones)

            integrated = put_status_bar(
                integrated,
                texts_colors=[
                    (now,                           (180,180,180)),
                    (f"모드: {mode_label}",          mode_color),
                    (f"사생활 보호: {p_text}"
                     f" | 블러: {blur_mode}",        p_color),
                    (f"얼굴구역: {face_cnt}개 | "
                     f"침대구역: {bed_cnt}개",        (200,200,200)),
                    (f"침대 상태: {m_text}",          m_color),
                    (f"움직임 점수: {score}",         m_color),
                    ("V키: 대시보드 | F/B키: 구역지정",(150,150,150)),
                ],
                start_y     = 10,
                font_size   = 20,
                line_height = 30
            )

            # ── 화면 출력 ───────────────────────────────
            if show_dashboard:
                display = make_dashboard(
                    frame, edge_result,
                    boundary_result, segment_result,
                    feature_result, integrated)
            else:
                display = integrated

            cv2.imshow("Baby Monitor", display)

        # ── 키보드 입력 ─────────────────────────────────
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("👋 종료합니다.")
            break

        elif key == ord(' '):
            paused = not paused
            print("⏸ 일시정지" if paused else "▶ 재생")

        elif key == ord('s'):
            filename = datetime.datetime.now().strftime(
                       "snapshot_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, display)
            print(f"📸 저장 완료: {filename}")

        elif key == ord('v'):
            show_dashboard = not show_dashboard
            print("📊 대시보드 "
                  f"{'ON' if show_dashboard else 'OFF'}")

        elif key == ord('p'):
            privacy_on = not privacy_on
            print(f"🔒 사생활 보호: "
                  f"{'ON' if privacy_on else 'OFF'}")

        elif key == ord('m'):
            blur_index = (blur_index+1) % len(BLUR_MODES)
            blur_mode  = BLUR_MODES[blur_index]
            print(f"🎨 블러 방식: {blur_mode}")

        else:
            handle_key(key)
except KeyboardInterrupt:
    # Ctrl+C 눌렀을 때 정상 종료
    print("\n⏹ 프로그램을 종료합니다.")

finally:
    # 항상 실행 — 리소스 정리
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 종료 완료")

cap.release()
cv2.destroyAllWindows()