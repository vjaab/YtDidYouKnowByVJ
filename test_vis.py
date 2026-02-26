import cv2
import numpy as np
import sys

def diagnose_text_visibility(video_path):
    cap = cv2.VideoCapture(video_path)
    issues = []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total}")
    for pos in [0.1, 0.3, 0.6, 0.9]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total*pos))
        ret, frame = cap.read()
        if ret:
            subtitle_region = frame[1250:1480, 90:990]
            brightness = np.mean(subtitle_region)
            header_region = frame[0:200, 0:1080]
            header_brightness = np.mean(header_region)
            issues.append({
                'position': pos,
                'subtitle_brightness': brightness,
                'header_brightness': header_brightness,
                'subtitle_has_dark_bg': brightness < 80,
                'header_has_dark_bg': header_brightness < 80
            })
    cap.release()
    return issues

if __name__ == "__main__":
    print(diagnose_text_visibility(sys.argv[1]))
