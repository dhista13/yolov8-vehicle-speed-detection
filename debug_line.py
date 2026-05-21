# untuk validasi kendaraan masuk garis

import cv2
from ultralytics import YOLO

VIDEO_PATH = 'videos/vehicle_detection1.mp4'
model = YOLO('yolov8s.pt')

# Sesuaikan dengan main.py
CY1 = 122
CY2 = 157
L1_X_KIRI  = 350
L1_X_KANAN = 641
L2_X_KIRI  = 298
L2_X_KANAN = 692
OFFSET = 30   

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    results = model(frame, conf=0.15, classes=[2,7], verbose=False)

    # Zona toleransi - warna putih tipis
    cv2.rectangle(frame,
                  (L1_X_KIRI, CY1-OFFSET), (L1_X_KANAN, CY1+OFFSET),
                  (255,255,255), 1)
    cv2.rectangle(frame,
                  (L2_X_KIRI, CY2-OFFSET), (L2_X_KANAN, CY2+OFFSET),
                  (255,255,255), 1)

    # Garis utama - PUTIH (sama dengan main.py)
    cv2.line(frame, (L1_X_KIRI,CY1), (L1_X_KANAN,CY1), (255,255,255), 2)
    cv2.putText(frame, f'L1 y={CY1}', (L1_X_KIRI, CY1-OFFSET-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.line(frame, (L2_X_KIRI,CY2), (L2_X_KANAN,CY2), (255,255,255), 2)
    cv2.putText(frame, f'L2 y={CY2}', (L2_X_KIRI, CY2-OFFSET-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Deteksi kendaraan
    if results[0].boxes is not None:
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        NAME    = {2:'car', 7:'truck'}
        COLOR   = {2:(0,0,255), 7:(0,128,0)}

        for i in range(len(boxes)):
            x1,y1,x2,y2 = map(int, boxes[i])
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            cls = classes[i]

            cv2.rectangle(frame,(x1,y1),(x2,y2), COLOR.get(cls,(200,200,0)), 2)
            cv2.circle(frame,(cx,cy),5,(255,255,255),-1)
            cv2.putText(frame, f"cy={cy}", (cx-20,cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
            cv2.putText(frame, NAME.get(cls,'?'), (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # Cek masuk zona
            if (CY1-OFFSET) < cy < (CY1+OFFSET):
                print(f"✅ MASUK L1! cy={cy}, CY1={CY1}")
                cv2.putText(frame, "HIT L1!", (cx,cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            if (CY2-OFFSET) < cy < (CY2+OFFSET):
                print(f"✅ MASUK L2! cy={cy}, CY2={CY2}")
                cv2.putText(frame, "HIT L2!", (cx,cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow('Cek Garis', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()