# step2_preprocess.py
import cv2
import numpy as np

def preprocess(frame, mode="night"):
    """
    mode = "night" : 야간 영상
    mode = "day"   : 주간 영상
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode == "night":
        denoised  = cv2.GaussianBlur(gray, (5, 5), 0)
        processed = cv2.equalizeHist(denoised)
    else:
        clahe     = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray)

    return gray, processed


def show_preprocess_result(frame, gray, processed):
    original_bgr  = frame.copy()
    gray_bgr      = cv2.cvtColor(gray,
                                  cv2.COLOR_GRAY2BGR)
    processed_bgr = cv2.cvtColor(processed,
                                  cv2.COLOR_GRAY2BGR)

    cv2.putText(original_bgr,  "Original",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(gray_bgr,      "Grayscale",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(processed_bgr, "Processed",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    return np.hstack([original_bgr,
                      gray_bgr,
                      processed_bgr])