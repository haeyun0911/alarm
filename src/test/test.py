import cv2
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image

# Haar Cascade XML 파일 경로 지정
face_cascade = cv2.CascadeClassifier('../../assets/haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier('../../assets/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('../../assets/haarcascade_eye.xml') # 눈 감지기 추가

font_path = "C:/Windows/Fonts/malgun.ttf"
font = ImageFont.truetype(font_path, 30)

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 얼굴 및 눈 감지 상태 기록용 변수
no_face_start = None
no_eye_start = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 정면 얼굴 감지
    faces_frontal = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    # 옆모습 얼굴 감지
    faces_profile = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    # 반전 이미지에서 옆모습 감지
    gray_flipped = cv2.flip(gray, 1)
    faces_profile_flipped = profile_cascade.detectMultiScale(gray_flipped, 1.1, 5, minSize=(30, 30))
    
    # 반전된 옆모습 좌표를 원래대로 변환
    faces_profile_flipped_corrected = []
    for (x, y, w, h) in faces_profile_flipped:
        faces_profile_flipped_corrected.append((gray.shape[1] - x - w, y, w, h))

    # 모든 감지 결과를 하나의 리스트로 합치기
    all_faces = list(faces_frontal) + list(faces_profile) + faces_profile_flipped_corrected
    
    is_face_detected = len(all_faces) > 0
    is_eyes_open = False
    
    if is_face_detected:
        no_face_start = None
        
        # 각 얼굴 영역에서 눈 감지
        for (x, y, w, h) in all_faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # 얼굴 영역(ROI)에서만 눈 감지
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            
            if len(eyes) >= 2:
                is_eyes_open = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
                
                text = "눈을 떴습니다. 알람 중지"
                color = (0, 255, 0)
                no_eye_start = None # 눈을 감지했으므로 타이머 초기화
                break
        
        if not is_eyes_open:
            # 얼굴은 있지만 눈을 감았거나 찾지 못함
            if no_eye_start is None:
                no_eye_start = time.time()
                
            elapsed = time.time() - no_eye_start
            
            if elapsed >= 3: # 예시: 3초 이상 눈을 감았을 경우
                text = "눈을 감았습니다. 알람 재시작"
                color = (0, 0, 255)
            else:
                remaining = int(3 - elapsed)
                text = f"{remaining}초 후 알람 재시작"
                color = (0, 165, 255)
    
    else:
        # 얼굴이 감지되지 않음
        if no_face_start is None:
            no_face_start = time.time()
        
        elapsed = time.time() - no_face_start
        if elapsed >= 5: # 5초 이상 얼굴이 없음
            text = "알람 재시작"
            color = (0, 0, 255)
        else:
            remaining = int(5 - elapsed)
            text = f"{remaining}초 후 알람 재시작"
            color = (0, 165, 255)
            
    # PIL로 한글 출력
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    draw.text((10, 10), text, font=font, fill=(color[2], color[1], color[0]))
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow('Face and Eye Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()