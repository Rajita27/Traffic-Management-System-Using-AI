import cv2
import os

# ==== CONFIG ====
video_path = 'Sample_Video_HighQuality.mp4'  # <-- change if your video has a different name
output_dir = 'dataset/images/all'            # folder where frames will be saved
frame_skip = 5                                # extract 1 frame every 5 frames
# =================

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        filename = os.path.join(output_dir, f"frame_{saved_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[+] Saved: {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done! Extracted {saved_count} frames.")
