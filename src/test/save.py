import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

# 1. YOLO 모델 로딩
try:
    model = YOLO("yolo11n.pt")  # yolov8n.pt 또는 다른 모델
except Exception as e:
    print(f"YOLO 모델 로딩 실패: {e}")
    exit()

# 2. Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 3. 한글 폰트 설정
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, 30)
except IOError:
    font = ImageFont.load_default()
    print("맑은 고딕 폰트를 찾을 수 없어 기본 폰트로 설정합니다.")

# 4. 상태 관리 변수
alarm_on = False
lying_start_time = None
ALARM_THRESHOLD_SECONDS = 5

# 5. 자세 판별 함수
def check_posture(landmarks):
    """랜드마크를 이용해 자세가 '누움'인지 '서있음'인지 판별"""
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # 4점 모두 가시성 0.6 초과일시 판별
    if not all(p.visibility > 0.6 for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return "unknown"

    # 좌우 어깨, 엉덩이의 중심점 계산
    shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2
    shoulder_center_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_center_x = (l_hip.x + r_hip.x) / 2
    hip_center_y = (l_hip.y + r_hip.y) / 2

    # 두 중심점의 수평거리와 수직거리
    dx = abs(shoulder_center_x - hip_center_x)
    dy = abs(shoulder_center_y - hip_center_y)

    # 수직 길이가 수평 길이의 1.5배 이상이면 '서있음'으로 판별
    is_upright = dy > dx * 1.5

    return "upright" if is_upright else "lying"

# 6. 비디오 캡처 시작
video_path = "../../assets/6.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

# 🎥 결과 영상 저장 준비 (리사이즈된 크기로 저장)
original_height, original_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
new_width = int(original_width / 3)
new_height = int(original_height / 3)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = "output_result.mp4"
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

# 7. 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("영상 끝에 도달했거나 파일을 읽을 수 없습니다. 프로그램을 종료합니다.")
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))
    results = model(resized_frame, verbose=False, classes=[0])  # 사람만 감지
    person_boxes = results[0].boxes.xyxy.cpu().numpy()

    status_text = "사람 없음"
    text_color = (0, 255, 0)

    # 한명 이상일시 가장 큰 박스를 대표 인물로 선택
    if len(person_boxes) > 0:
        main_person_box = max(person_boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        x1, y1, x2, y2 = map(int, main_person_box)

        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        posture = "unknown"
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                resized_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
            posture = check_posture(pose_results.pose_landmarks.landmark)

        if posture == "lying":
            if lying_start_time is None:
                lying_start_time = time.time()
            else:
                lying_duration = time.time() - lying_start_time
                remaining_time = ALARM_THRESHOLD_SECONDS - lying_duration
                if remaining_time > 0:
                    status_text = f"누워있는 중... {int(remaining_time)+1}초 후 알람"
                    text_color = (0, 165, 255)
                else:
                    alarm_on = True
        elif posture == "upright":
            lying_start_time = None
            alarm_on = False
            status_text = "정상 상태"
            text_color = (0, 255, 0)
        else:
            status_text = "자세 인식 불가"
            text_color = (255, 255, 0)

        if alarm_on:
            status_text = f"알람! {ALARM_THRESHOLD_SECONDS}초 이상 누워있습니다!"
            text_color = (0, 0, 255)
    else:
        alarm_on = False
        lying_start_time = None

    # PIL을 이용한 한글 텍스트 표시
    frame_pil = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((20, 20), status_text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    resized_frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    # 결과 출력 & 저장
    cv2.imshow("Lying Detection System", resized_frame)
    out.write(resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

# 8. 리소스 해제
cap.release()
out.release()
cv2.destroyAllWindows()
pose.close()
