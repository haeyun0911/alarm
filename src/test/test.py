import cv2

# Haar Cascade XML 파일 경로 지정
face_cascade = cv2.CascadeClassifier('../../assets/haarcascade_frontalface_default.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break
    
    # 흑백 이미지로 변환 (얼굴 감지는 흑백 이미지에서 더 효과적입니다)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 감지
    # scaleFactor: 이미지 크기 축소 비율, minNeighbors: 얼굴 후보 지점의 최소 개수
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 감지된 얼굴에 사각형 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        text = "stop alarm"
        
        org = (10, 30)  # 텍스트 출력 위치 (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 255, 0) 
        thickness = 2
            
        # 결과 화면에 표시
        cv2.putText(frame, text, org, font, font_scale, color, thickness)
    cv2.imshow('Face Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 시 자원 해제
cap.release()
cv2.destroyAllWindows()