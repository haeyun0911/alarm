import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image


try:
    model = YOLO("yolo11n.pt")
except Exception as e:
    print(f"YOLO 모델 로딩 실패: {e}")


# -----------------------------
# 2. Mediapipe Pose 초기화
# -----------------------------
mp_pose = mp.solutions.pose
# 감지 신뢰도와 추적 신뢰도를 조절하여 정확도를 높일 수 있습니다.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# 3. 한글 폰트 설정 (PIL)
# -----------------------------
# 폰트 경로는 시스템에 맞게 설정해야 합니다.
# Windows의 기본 '맑은 고딕' 폰트를 사용합니다.
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, 30)
except IOError:
    print(f"폰트 파일을 찾을 수 없습니다: {font_path}")
    print("다른 경로의 폰트 파일을 지정하거나, 시스템에 맞는 폰트 경로로 수정해주세요.")
    # 폰트가 없어도 프로그램이 동작하도록 기본 폰트로 설정 (영문만 가능)
    font = ImageFont.load_default()


# -----------------------------
# 4. 상태 관리 변수
# -----------------------------
alarm_on = False
lying_start_time = None
ALARM_THRESHOLD_SECONDS = 5 # 알람 울릴 시간 (초)

# -----------------------------
# 5. 자세 판별 함수 (개선)
# -----------------------------
def get_slope(p1, p2):
    """두 점 사이의 기울기를 계산합니다."""
    # p2.x와 p1.x가 같으면 ZeroDivisionError 방지를 위해 무한대 반환
    if p2.x - p1.x == 0:
        return float("inf")
    return (p2.y - p1.y) / (p2.x - p1.x)

def check_posture(landmarks):
    """랜드마크를 이용해 자세가 '누움'인지 '서있음'인지 판별합니다."""
    # 필요한 랜드마크 인덱스
    l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # 랜드마크가 화면에 보이는지 확인 (visibility)
    if not all(p.visibility > 0.5 for p in [l_shoulder, r_shoulder, l_hip, r_hip]):
        return "unknown" # 주요 부위가 보이지 않으면 판별 불가

    # 어깨와 엉덩이의 기울기 계산
    shoulder_slope = abs(get_slope(l_shoulder, r_shoulder))
    hip_slope = abs(get_slope(l_hip, r_hip))

    # 어깨와 엉덩이의 수직 거리 계산 (상체 길이)
    # y좌표는 아래로 갈수록 커지므로 abs 사용
    torso_height = abs(((l_shoulder.y + r_shoulder.y) / 2) - ((l_hip.y + r_hip.y) / 2))
    # 어깨 너비
    shoulder_width = abs(l_shoulder.x - r_shoulder.x)

    # 누워있는 자세 판별 조건 강화
    # 1. 어깨와 엉덩이 기울기가 수평에 가까운가? (기존 로직)
    # 2. 상체의 수직 길이가 어깨 너비보다 짧은가? (누우면 상체가 짧아 보임)
    #    (단, 어깨 너비가 0인 경우 방지)
    is_lying = shoulder_slope < 0.5 and hip_slope < 0.5
    if shoulder_width > 0: # 0으로 나누는 것 방지
        is_lying = is_lying and (torso_height / shoulder_width < 1.0)

    if is_lying:
        return "lying"
    else:
        return "upright"

# -----------------------------
# 6. 비디오 캡처 시작
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    h, w, _ = frame.shape

    # YOLO: 사람 감지
    results = model(frame, verbose=False, classes=[0]) # class=0 (person)만 감지
    person_boxes = results[0].boxes.xyxy.cpu().numpy()

    # 화면에 표시할 텍스트와 색상 초기화
    status_text = "사람 없음"
    text_color = (0, 255, 0) # BGR: Green

    if len(person_boxes) > 0:
        # 가장 큰 바운딩 박스를 가진 사람을 기준으로 처리
        main_person_box = max(person_boxes, key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
        x1, y1, x2, y2 = map(int, main_person_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # 사람 영역 표시

        # Mediapipe Pose: 관절 추출 (BGR -> RGB 변환 필요)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_frame)

        posture = "unknown"
        if pose_results.pose_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
            posture = check_posture(pose_results.pose_landmarks.landmark)

        # -----------------------------
        # 7. 알람 로직 재구성
        # -----------------------------
        if posture == "lying":
            if lying_start_time is None:
                # 누운 상태가 처음 감지되면 시간 기록
                lying_start_time = time.time()
            else:
                # 누운 상태가 지속된 시간 계산
                lying_duration = time.time() - lying_start_time
                remaining_time = ALARM_THRESHOLD_SECONDS - lying_duration
                
                if remaining_time > 0:
                    status_text = f"누워있는 중... {int(remaining_time)+1}초 후 알람"
                    text_color = (0, 165, 255) # BGR: Orange
                else:
                    alarm_on = True # 알람 활성화

        elif posture == "upright":
            # 서 있는 상태가 감지되면 알람 관련 변수 모두 초기화
            lying_start_time = None
            alarm_on = False
            status_text = "정상 상태"
            text_color = (0, 255, 0) # BGR: Green
        else: # "unknown"
            status_text = "자세 인식 불가"
            text_color = (255, 255, 0) # BGR: Cyan
            # 알 수 없는 상태에서는 타이머를 리셋하지 않아, 잠시 가려져도 유예를 둠

        # 알람이 활성화된 경우
        if alarm_on:
            status_text = f"알람! {ALARM_THRESHOLD_SECONDS}초 이상 누워있습니다!"
            text_color = (0, 0, 255) # BGR: Red
            # (옵션) 여기에 소리 알람 등을 추가할 수 있습니다.
            # import winsound
            # winsound.Beep(1000, 500) # 1000Hz 소리를 0.5초간

    else: # 사람이 감지되지 않은 경우
        alarm_on = False
        lying_start_time = None


    # -----------------------------
    # 8. 화면에 한글 텍스트 출력 (PIL)
    # -----------------------------
    # OpenCV 프레임(BGR)을 PIL 이미지(RGB)로 변환
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    # PIL의 fill 색상은 RGB 순서이므로 BGR을 RGB로 변환
    draw.text((20, 20), status_text, font=font, fill=(text_color[2], text_color[1], text_color[0]))
    # 다시 PIL 이미지를 OpenCV 프레임으로 변환
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)


    cv2.imshow("Lying Detection System", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

