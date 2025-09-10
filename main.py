# facial_landmarks_final_relative.py
from imutils import face_utils
import cv2
import dlib
import os

def detect_landmarks(image_path, predictor_path, output_path="츄.jpg", target_width=500):
    # dlib 얼굴 검출기 + 랜드마크 예측기 초기화
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 찾을 수 없습니다:", image_path)
        return

    # 리사이즈
    scale = target_width / image.shape[1]
    new_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (target_width, new_height))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    rects = detector(gray, 1)
    print(f"감지된 얼굴 수: {len(rects)}")

    for idx, rect in enumerate(rects):
        # 랜드마크 추출
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 얼굴 박스
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Face #{idx+1}", (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 랜드마크 점 찍기
        for (lx, ly) in shape:
            cv2.circle(image, (lx, ly), 3, (0, 0, 255), -1)  # 점 크기 3으로 조정

    # 결과 저장
    cv2.imwrite(output_path, image)
    print(f"랜드마크 감지 완료: {output_path}")

# ----------------------
# 바로 실행 가능
# ----------------------
if __name__ == "__main__":
    # 상대경로 기준 (스크립트와 같은 폴더)
    predictor_path = "./model/shape_predictor_68_face_landmarks.dat"
    image_path = "./use_image/츄2.png"
    output_path = "./output_image/dlib_landmarks_final_relative_츄2.png"

    detect_landmarks(image_path, predictor_path, output_path)
