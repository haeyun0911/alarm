import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
import math

# ----------------------------
# 모델 초기화
# ----------------------------
model = YOLO("yolo11n.pt")  # 경량 YOLO 모델
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ----------------------------
# 카메라 열기
# ----------------------------
cap = cv2.VideoCapture(0)

# ----------------------------
# 알람 관련 변수
# ----------------------------
alarm_active = False
eye_closed_start = None
lying_start = None

# ----------------------------
# YOLO 최적화 변수
# ----------------------------
frame_count = 0
yolo_interval = 5   # YOLO 실행 간격 (5프레임마다 실행)
last_box = None     # 마지막 사람 영역 저장

# ----------------------------
# 눈 감김 상태 체크 (예시: 간단히 면적 비율 활용)
# ----------------------------
def is_eye_closed(landmarks):
    # 여기서는 간단히 "눈 감김 여부"를 랜덤으로 True/False로 판단한다고 가정
    # 실제 적용하려면 Mediapipe FaceMesh 추가 필요
    return False  

# ----------------------------
# 기울기로 누워있는지 판단
# ----------------------------
def is_lying_pose(landmarks, img_w, img_h):
    try:
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 좌표 변환
        l_sh = (int(l_sh.x * img_w), int(l_sh.y * img_h))
        r_sh = (int(r_sh.x * img_w), int(r_sh.y * img_h))
        l_hip = (int(l_hip.x * img_w), int(l_hip.y * img_h))
        r_hip = (int(r_hip.x * img_w), int(r_hip.y * img_h))

        # 어깨선, 엉덩이선 기울기 (radian → degree)
        shoulder_angle = math.degrees(math.atan2(r_sh[1]-l_sh[1], r_sh[0]-l_sh[0]))
        hip_angle = math.degrees(math.atan2(r_hip[1]-l_hip[1], r_hip[0]-l_hip[0]))

        # 수평에 가까우면 누움
        if abs(shoulder_angle) < 20 and abs(hip_angle) < 20:
            return True
    except:
        pass
    return False

# ----------------------------
# 메인 루프
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    img_h, img_w, _ = frame.shape
    person_box = None

    # ---------------- YOLO (5프레임마다 실행) ----------------
    if frame_count % yolo_interval == 0:
        results = model(frame, verbose=False)
        biggest_person = None
        max_area = 0
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2-x1)*(y2-y1)
                    if area > max_area:
                        max_area = area
                        biggest_person = (x1, y1, x2, y2)
        last_box = biggest_person

    person_box = last_box

    # ---------------- Pose & 알람 체크 ----------------
    if person_box:
        x1, y1, x2, y2 = person_box
        roi = frame[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result_pose = pose.process(rgb_roi)

        if result_pose.pose_landmarks:
            landmarks = result_pose.pose_landmarks.landmark
            lying = is_lying_pose(landmarks, roi.shape[1], roi.shape[0])
            eyes_closed = is_eye_closed(landmarks)

            if eyes_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elif time.time() - eye_closed_start > 5:
                    alarm_active = True
            else:
                eye_closed_start = None

            if lying:
                if lying_start is None:
                    lying_start = time.time()
                elif time.time() - lying_start > 5:
                    alarm_active = True
            else:
                lying_start = None

            # 해제 조건
            if not lying and not eyes_closed:
                alarm_active = False

        # 박스 그리기
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    else:
        # 사람 없으면 알람 해제
        alarm_active = False

    # ---------------- 상태 표시 ----------------
    status = "ALARM!" if alarm_active else "SAFE"
    color = (0,0,255) if alarm_active else (0,255,0)
    cv2.putText(frame, status, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("Alarm System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
