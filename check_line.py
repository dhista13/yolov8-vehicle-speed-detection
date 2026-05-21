# untuk menentukan posisi garis L1, L2, dan zona toleransi secara visual

import cv2
from ultralytics import YOLO

VIDEO_PATH = 'videos/vehicle_detection1.mp4'
model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 4)
ret, frame = cap.read()
cap.release()

frame = cv2.resize(frame, (1020, 500))

# Deteksi kendaraan untuk referensi posisi
hasil = model(frame.copy(), conf=0.15, classes=[2,7], verbose=False)
if hasil[0].boxes is not None:
    for i, box in enumerate(hasil[0].boxes.xyxy.cpu().numpy()):
        x1,y1,x2,y2 = map(int,box)
        cy = (y1+y2)//2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"cy={cy}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1)

titik = []

def klik(event, x, y,):
    if event == cv2.EVENT_LBUTTONDOWN:
        titik.append((x,y))
        cv2.circle(frame,(x,y),6,(0,255,255),-1)
        cv2.putText(frame,f"{len(titik)}:({x},{y})",
                    (x+5,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,255),1)

        # Gambar garis L1 setelah klik ke-2 - PUTIH
        if len(titik)==2:
            cv2.line(frame,titik[0],titik[1],(255,255,255),2)
            cy1=(titik[0][1]+titik[1][1])//2
            cv2.putText(frame,f"L1 CY={cy1}",
                        (titik[0][0],titik[0][1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            print(f"\n L1: CY1={cy1}")
            print(f"   L1_X_KIRI={min(titik[0][0],titik[1][0])}")
            print(f"   L1_X_KANAN={max(titik[0][0],titik[1][0])}")

        # Gambar garis L2 setelah klik ke-4 - PUTIH
        if len(titik)==4:
            cv2.line(frame,titik[2],titik[3],(255,255,255),2)
            cy2=(titik[2][1]+titik[3][1])//2
            cv2.putText(frame,f"L2 CY={cy2}",
                        (titik[2][0],titik[2][1]-15),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            print(f"\n L2: CY2={cy2}")
            print(f"   L2_X_KIRI={min(titik[2][0],titik[3][0])}")
            print(f"   L2_X_KANAN={max(titik[2][0],titik[3][0])}")
            print(f"\n SALIN KE main.py:")
            print(f"   CY1 = {(titik[0][1]+titik[1][1])//2}")
            print(f"   CY2 = {cy2}")
            print(f"   L1_X_KIRI  = {min(titik[0][0],titik[1][0])}")
            print(f"   L1_X_KANAN = {max(titik[0][0],titik[1][0])}")
            print(f"   L2_X_KIRI  = {min(titik[2][0],titik[3][0])}")
            print(f"   L2_X_KANAN = {max(titik[2][0],titik[3][0])}")

cv2.namedWindow('Klik 4 Titik Garis')
cv2.setMouseCallback('Klik 4 Titik Garis', klik)

print("="*50)
print("Klik 1 → ujung KIRI  garis L1")
print("Klik 2 → ujung KANAN garis L1")
print("Klik 3 → ujung KIRI  garis L2")
print("Klik 4 → ujung KANAN garis L2")
print("ESC = keluar & lihat hasil di terminal")
print("="*50)

while True:
    cv2.imshow('Klik 4 Titik Garis', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()