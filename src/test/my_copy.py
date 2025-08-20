import os
import shutil

# --------------------------------------------------
# ✨ 여기를 수정해주세요! ✨
source_directory = "C:/Users/405/Downloads/사람 동작 영상(2020)/Training/[원천]기본동작_눕기_TR"
destination_directory = "C:/Users/405/projects/miniproject/alarm/assets/copy"
# --------------------------------------------------

image_extensions = ('.jpg', '.jpeg', '.png') # 확장자 목록을 더 넓게 설정

os.makedirs(destination_directory, exist_ok=True)

# os.walk를 사용하여 모든 하위 폴더를 탐색합니다.
for dirpath, dirnames, filenames in os.walk(source_directory):
    # dirpath: 현재 폴더 경로
    # dirnames: 현재 폴더 안에 있는 하위 폴더 목록
    # filenames: 현재 폴더 안에 있는 파일 목록

    print(f"[{os.path.basename(dirpath)}] 폴더를 확인하는 중...")

    # 이미지 파일만 필터링
    image_files = [f for f in filenames if f.lower().endswith(image_extensions)]

    if not image_files:
        print("  이미지 파일이 없습니다.")
        continue

    image_files.sort()
    last_five_images = image_files[-5:]

    for image_name in last_five_images:
        source_path = os.path.join(dirpath, image_name)
        
        # 폴더명을 파일명 앞에 붙여서 중복 방지
        parent_folder_name = os.path.basename(dirpath)
        new_filename = f"{parent_folder_name}_{image_name}"
        destination_path = os.path.join(destination_directory, new_filename)

        shutil.copy2(source_path, destination_path)
        print(f"  -> 복사 완료: {new_filename}")

print("\n🎉 모든 작업이 완료되었습니다!")