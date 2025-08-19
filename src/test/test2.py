from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

face_cascade = cv2.CascadeClassifier('../../assets/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('../../assets/haarcascade_profileface.xml')

font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 30)
# YOLO 모델 불러오기
model = YOLO("yolo11n.pt")  # 필요시 yolo11s/m.pt로 변경

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

def get_angle(p1, p2):
    """두 점을 연결한 선의 기울기 각도(도 단위)"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return abs(angle)  # 0도(수평), 90도(수직)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # YOLO 탐지
    text = "사람 없음"
    color = (0, 0, 255)

    biggest_person = None
    max_area = 0

    # 사람 중 가장 큰 박스 선택
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    biggest_person = (x1, y1, x2, y2)

    if biggest_person:
        x1, y1, x2, y2 = biggest_person
        person_roi = frame[y1:y2, x1:x2]

        # Mediapipe Pose 적용
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        result_pose = pose.process(rgb_roi)

        if result_pose.pose_landmarks:
            h, w, _ = person_roi.shape
            lm = result_pose.pose_landmarks.landmark

            # 어깨, 엉덩이 좌표 추출
            l_shoulder = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                          int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            r_shoulder = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                          int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
            l_hip = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                     int(lm[mp_pose.PoseLandmark.LEFT_HIP].y * h))
            r_hip = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

            # 어깨선과 엉덩이선 기울기 계산
            shoulder_angle = get_angle(l_shoulder, r_shoulder)
            hip_angle = get_angle(l_hip, r_hip)
            avg_angle = (shoulder_angle + hip_angle) / 2

            if avg_angle < 30:   # 수평 → 누워있음
                text = "누워있음 → 알람 시작"
                color = (0, 0, 255)
            else:  # 수직 → 서있음/앉음
                text = "서있음/앉음 → 알람 중지"
                color = (0, 255, 0)

            # 시각화
            cv2.circle(person_roi, l_shoulder, 5, (255, 0, 0), -1)
            cv2.circle(person_roi, r_shoulder, 5, (255, 0, 0), -1)
            cv2.circle(person_roi, l_hip, 5, (0, 255, 0), -1)
            cv2.circle(person_roi, r_hip, 5, (0, 255, 0), -1)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # PIL로 한글 출력
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10, 10), text, font=font, fill=(color[2], color[1], color[0]))
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLO + Mediapipe Pose (1 person)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
