"""
UCSD Ped2 .tif 시퀀스 -> 데모용 동영상(.mp4) 변환 스크립트
============================================================

발표 시연용으로 UCSD Test 영상을 .mp4로 변환합니다.
변환된 영상은 anomaly_detection.py의 --video 옵션으로 바로 사용 가능합니다.

사용법:
  1. 아래 TEST_FOLDER, OUTPUT_PATH 를 환경에 맞게 수정
  2. python convert_ucsd_to_video.py 실행
  3. 생성된 .mp4 파일을 anomaly_detection.py --video 로 사용
"""

import cv2
import glob
import os

# ─── 설정 ────────────────────────────────────────────────────────────────────
TEST_FOLDER = r"C:\Users\IBYEO\PycharmProjects\grad_school\course_cv\src\project\final\data\UCSDped2\Test\Test010" # 변환할 영상 폴더 (Test004 추천)
OUTPUT_PATH = r"demo_test010.mp4"        # 출력 파일명

FPS = 10          # 출력 동영상의 초당 프레임 수 (10~15 권장)
SCALE = 2         # 화면 확대 비율 (원본이 작아서 UI 오버레이가 잘 보이도록 확대)


def convert(test_folder, output_path, fps=10, scale=2):
    files = sorted(glob.glob(os.path.join(test_folder, "*.tif")))
    if not files:
        print(f"[ERROR] .tif 파일을 찾을 수 없습니다: {test_folder}")
        return

    # 첫 프레임으로 크기 확인
    first = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    h, w = first.shape
    out_w, out_h = w * scale, h * scale

    print(f"[INFO] 원본 크기: {w}x{h} -> 출력 크기: {out_w}x{out_h}")
    print(f"[INFO] 총 프레임 수: {len(files)}, FPS: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    for f in files:
        gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if scale != 1:
            gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.write(bgr)

    out.release()
    print(f"[완료] {output_path} 생성됨 ({len(files)} 프레임)")


if __name__ == "__main__":
    convert(TEST_FOLDER, OUTPUT_PATH, fps=FPS, scale=SCALE)