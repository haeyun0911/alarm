import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
import mediapipe as mp

# Haar Cascade
face_cascade = cv2.CascadeClassifier('../../assets/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('../../assets/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('../../assets/haarcascade_eye.xml')

# 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 30)

# YOLO + Mediapipe
model = YOLO("yolo11n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

# 웹캠
cap = cv2.VideoCapture(0)

# 상태 타이머
eye_closed_start = None
lying_start = None

frame_count = 0
yolo_interval = 5   # 5프레임마다 YOLO 실행
last_box = None     # 이전 YOLO 결과 저장

def get_angle(p1, p2):
    dx = p2[0]-p1[0]
    dy = p2[1]-p1[1]
    return abs(np.degrees(np.arctan2(dy, dx)))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar 얼굴/눈 감지
    faces_frontal = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    faces_profile = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    gray_flipped = cv2.flip(gray,1)
    faces_profile_flipped = profile_cascade.detectMultiScale(gray_flipped, 1.1, 5, minSize=(30,30))
    faces_profile_flipped_corrected = [(gray.shape[1]-x-w, y, w, h) for (x,y,w,h) in faces_profile_flipped]

    all_faces = list(faces_frontal) + list(faces_profile) + faces_profile_flipped_corrected
    is_face_detected = len(all_faces) > 0
    is_eyes_open = False

    frame_count += 1

    if frame_count % yolo_interval == 0:
        # YOLO 실행
        results = model(frame, verbose=False)
        biggest_person = None
        max_area = 0
        # 감지된 사람 중 가장 큰 box 선택
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # person
                    biggest_person = box.xyxy[0].cpu().numpy().astype(int)
                    last_box = biggest_person
    else:
        # YOLO 건너뛰고 이전 박스 사용
        biggest_person = last_box

    if biggest_person is not None:
        x1, y1, x2, y2 = biggest_person
        person_roi = frame[y1:y2, x1:x2]

    # 눈 감지
    if is_face_detected:
        for (x,y,w,h) in all_faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5)
            if len(eyes) >= 2:
                is_eyes_open = True
                eye_closed_start = None
                break
        if not is_eyes_open:
            if eye_closed_start is None:
                eye_closed_start = time.time()
    else:
        eye_closed_start = None

    # YOLO+Pose로 누움 상태 확인
    biggest_person = None
    max_area = 0
    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls]=="person":
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                area = (x2-x1)*(y2-y1)
                if area>max_area:
                    max_area = area
                    biggest_person = (x1,y1,x2,y2)

    is_lying = False
    if biggest_person:
        x1,y1,x2,y2 = biggest_person
        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result_pose = pose.process(rgb_roi)
        if result_pose.pose_landmarks:
            h,w,_ = roi.shape
            lm = result_pose.pose_landmarks.landmark
            l_sh = (int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x*w),
                    int(lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y*h))
            r_sh = (int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x*w),
                    int(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y*h))
            l_hip = (int(lm[mp_pose.PoseLandmark.LEFT_HIP].x*w),
                     int(lm[mp_pose.PoseLandmark.LEFT_HIP].y*h))
            r_hip = (int(lm[mp_pose.PoseLandmark.RIGHT_HIP].x*w),
                     int(lm[mp_pose.PoseLandmark.RIGHT_HIP].y*h))
            shoulder_angle = get_angle(l_sh,r_sh)
            hip_angle = get_angle(l_hip,r_hip)
            avg_angle = (shoulder_angle+hip_angle)/2
            if avg_angle < 30:  # 수평 → 누움
                is_lying = True
                if lying_start is None:
                    lying_start = time.time()
            else:
                lying_start = None

    # 알람 상태 판단
    alarm_active = False
    current_time = time.time()
    eye_closed_elapsed = current_time - eye_closed_start if eye_closed_start else 0
    lying_elapsed = current_time - lying_start if lying_start else 0

    if (is_face_detected and (eye_closed_elapsed >= 5 or lying_elapsed >= 5)):
        alarm_active = True
        text = "알람 활성화!"
        color = (0,0,255)
    else:
        alarm_active = False
        text = "알람 해제"
        color = (0,255,0)

    # 아무것도 감지되지 않은 경우 알람 해제
    if not is_face_detected and biggest_person is None:
        alarm_active = False
        text = "알람 해제 (감지 없음)"
        color = (0,255,0)

    # 시각화
    if biggest_person:
        cv2.rectangle(frame, (biggest_person[0], biggest_person[1]),
                      (biggest_person[2], biggest_person[3]), (0,255,0),2)
    for (x,y,w,h) in all_faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10,10), text, font=font, fill=(color[2],color[1],color[0]))
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("알람 감지", frame)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
