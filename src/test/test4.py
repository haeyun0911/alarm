import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# -----------------------------
# YOLO 모델 (사람 탐지 전용)
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# Mediapipe Pose 초기화
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# -----------------------------
# 한글 폰트 설정 (PIL)
# -----------------------------
font_path = "C:/Windows/Fonts/malgun.ttf"  # 경로 맞게 수정
font = ImageFont.truetype(font_path, 30)

# -----------------------------
# 알람 상태 관리 변수
# -----------------------------
alarm_on = False
lying_start = None

# -----------------------------
# 자세 판별 함수
# -----------------------------
def get_slope(p1, p2):
    if p2[0] - p1[0] == 0:
        return float("inf")
    return (p2[1] - p1[1]) / (p2[0] - p1[0])

def check_posture(landmarks, w, h):
    # 어깨, 엉덩이 좌표
    l_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                  int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    r_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                  int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    l_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
             int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))
    r_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w),
             int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h))

    # 기울기 계산
    shoulder_slope = abs(get_slope(l_shoulder, r_shoulder))
    hip_slope = abs(get_slope(l_hip, r_hip))

    # 수평에 가까우면 누워있음
    if shoulder_slope < 0.3 and hip_slope < 0.3:
        return "lying"
    else:
        return "upright"

# -----------------------------
# 비디오 캡처
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    # YOLO: 사람 감지
    results = model(frame, verbose=False)[0]
    persons = [box for box in results.boxes if int(box.cls[0]) == 0]  # class=0 → person

    text = "사람 없음"
    color = (0, 255, 0)

    if persons:
        # 첫 번째 사람만 추적
        x1, y1, x2, y2 = map(int, persons[0].xyxy[0])
        roi = frame[y1:y2, x1:x2]

        # Haar Cascade: 얼굴 & 눈
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Mediapipe Pose: 관절 추출
        results_pose = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        posture = None
        if results_pose.pose_landmarks:
            posture = check_posture(results_pose.pose_landmarks.landmark, w, h)

        # -----------------------------
        # 알람 조건 처리
        # -----------------------------
        

        if posture == "lying":
            if lying_start is None:
                lying_start = time.time()
            elif time.time() - lying_start >= 5:
                alarm_on = True
                text = "알람: 누워있음"
                color = (0, 0, 255)
        else:
            lying_start = None

        # 알람 해제 조건
        if (posture != "lying"):
            alarm_on = False
            text = "정상 상태"
            color = (0, 255, 0)

    # -----------------------------
    # 화면에 한글 출력 (PIL)
    # -----------------------------
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10, 10), text, font=font, fill=(color[2], color[1], color[0]))
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Alarm System", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
