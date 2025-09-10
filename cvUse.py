import cv2

img_path = 'testImage1.jpg'
img = cv2.imread(img_path)
if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OpenCV 기본 Haar Cascade 얼굴 탐지
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

output_path = 'opencv_detected.jpg'
cv2.imwrite(output_path, img)
print(f"얼굴 감지 완료: {output_path}")
