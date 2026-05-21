import cv2
import numpy as np
from ultralytics import YOLO
import time

# KONFIGURASI FILE 
VIDEO_PATH  = 'videos/vehicle_detection1.mp4'
OUTPUT_PATH = 'output/output_results1.mp4'

# POSISI GARIS (salin dari check_line.py)
CY1 = 122
CY2 = 157
L1_X_KIRI  = 350
L1_X_KANAN = 641
L2_X_KIRI  = 298
L2_X_KANAN = 692

# PARAMETER
JARAK_METER = 10    #jarak nyata antar garis dalam meter
OFFSET      = 30    #area toleransi untuk deteksi masuk garis (dalam pixel)

# sesuai dengan indeks kelas pada model YOLO (0=person, 1=bicycle, 2=car, 3=motorcycle, 4=airplane, 5=bus, 6=train, 7=truck, dst)
KELAS_VALID = {2:'car', 7:'truck'}
# Warna untuk bounding box berdasarkan kelas
COLOR = {
    2: (0, 0, 255),
    7: (0, 128, 0)
}

# INISIALISASI MODEL
model  = YOLO('yolov8s.pt')

# MEMBUKA VIDEO SESUAI PATH
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Video tidak bisa dibuka!")
    exit()

# INFO VIDEO
fps_video    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps_video:.1f} FPS | {total_frames} frames")

# KONFIGURASI OUTPUT VIDEO
# FPS output dibuat sama dengan video asli supaya durasi output = durasi asli, tapi frame diproses hanya setengahnya supaya lebih cepat
fps_out    = fps_video
fps_proses = fps_video / 2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_out, (1020, 500))

# Kecepatan hanya tampil 1 detik saja
DURASI_TAMPIL_FRAME = int(
    fps_proses * 1.0
)

# ID MAPPER & TRACKING DATA
id_map = {}
id_counter = 0
def get_id_rapi(id_asli):
    global id_counter
    if id_asli not in id_map:
        id_counter += 1
        id_map[id_asli] = id_counter
    return id_map[id_asli]

# DATA UNTUK MENGHITUNG KECEPATAN
frame_in_L1     = {}
frame_in_L2     = {}
sudah_hitung_down = set()
sudah_hitung_up  = set()
counter_down      = []
counter_up      = []
info_speed    = {}

# History digunakan untuk mendeteksi arah kendaraan (naik atau turun)
riwayat_cy  = {}
HISTORY_LEN = 5  

# FUNGSI BANTU untuk cek apakah suatu titik (cy) berada dalam garis deteksi
def in_line(cy, cy_garis, offset):
    return (cy_garis - offset) < cy < (cy_garis + offset)

# FUNGSI BANTU untuk mendeteksi arah kendaraan berdasarkan perubahan posisi cy
def deteksi_arah(obj_id):
    if obj_id not in riwayat_cy:
        return None
    history = riwayat_cy[obj_id]
    if len(history) < 3:
        return None
    changed = [history[j] - history[j-1] for j in range(1, len(history))]
    avg  = sum(changed) / len(changed)
    if avg > 0.5:
        return 'down'
    elif avg < -0.5:
        return 'up'
    return None

# MAIN LOOP
frame_count    = 0
frame_diproses = 0
start_time    = time.time()

print("\n Memproses video... ESC untuk berhenti\n")

# Baca frame per frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame_diproses += 1
    frame = cv2.resize(frame, (1020, 500))

    # YOLO TRACKING dengan botsort (ID lebih stabil dibanding bytetrack)
    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.20,
        iou=0.50,
        imgsz=640,
        classes=list(KELAS_VALID.keys()),
        verbose=False
    )

    # gambar garis pendeteksi kecepatan
    cv2.line(frame, (L1_X_KIRI, CY1), (L1_X_KANAN, CY1), (255,255,255), 2)
    cv2.putText(frame, 'L1', (L1_X_KIRI, CY1-8),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)

    cv2.line(frame, (L2_X_KIRI, CY2), (L2_X_KANAN, CY2), (255,255,255), 2)
    cv2.putText(frame, 'L2', (L2_X_KIRI, CY2-8),
                cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255), 2)

    # Proses deteksi dan tracking
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes   = results[0].boxes.xyxy.cpu().numpy()
        ids     = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        # Loop untuk setiap objek terdeteksi
        for i in range(len(ids)):
            cls = classes[i]
            if cls not in KELAS_VALID:
                continue

            # Ambil koordinat bounding box
            x1,y1,x2,y2 = map(int, boxes[i])
            obj_id      = ids[i]
            obj_id_rapi = get_id_rapi(obj_id)
            name        = KELAS_VALID[cls]
            color       = COLOR.get(cls, (200,200,0))

            # Titik tengah kendaraan
            cx = (x1+x2)//2
            cy = (y1+y2)//2

            # Update riwayat cy / posisi y
            if obj_id not in riwayat_cy:
                riwayat_cy[obj_id] = []
            riwayat_cy[obj_id].append(cy)
            if len(riwayat_cy[obj_id]) > HISTORY_LEN:
                riwayat_cy[obj_id].pop(0)
            
            arah = deteksi_arah(obj_id)     # deteksi arah berdasarkan perubahan posisi cy

            # Gambar bounding box, ID, kelas, dan arah
            cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
            cv2.circle(frame,(cx,cy),4,(255,255,255),-1)

            simbol = '↓' if arah=='down' else ('↑' if arah=='up' else '?')
            cv2.putText(frame, f"{obj_id_rapi}:{name[:3]}{simbol}",
                        (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # ARAH TURUN: L1 → L2 
            if arah == 'down':
                if in_line(cy, CY1, OFFSET):
                    if obj_id not in frame_in_L1:
                        frame_in_L1[obj_id] = frame_diproses

                if obj_id in frame_in_L1 and obj_id not in sudah_hitung_down:
                    if in_line(cy, CY2, OFFSET):
                        df = frame_diproses - frame_in_L1[obj_id]
                        dt = df / fps_proses
                        if 0.3 < dt < 30:
                            spd = (JARAK_METER / dt) * 3.6
                            info_speed[obj_id] = {
                                'kmh'          : spd,
                                'frame_deteksi': frame_diproses
                            }
                            sudah_hitung_down.add(obj_id)
                            counter_down.append(obj_id)
                            print(f"⬇️  ID {obj_id_rapi:>3} ({name:<5}) | {spd:>6.1f} km/h")

            # ARAH NAIK: L2 → L1
            elif arah == 'up':
                if in_line(cy, CY2, OFFSET):
                    if obj_id not in frame_in_L2:
                        frame_in_L2[obj_id] = frame_diproses

                if obj_id in frame_in_L2 and obj_id not in sudah_hitung_up:
                    if in_line(cy, CY1, OFFSET):
                        df = frame_diproses - frame_in_L2[obj_id]
                        dt = df / fps_proses
                        if 0.3 < dt < 30:
                            spd = (JARAK_METER / dt) * 3.6
                            info_speed[obj_id] = {
                                'kmh'          : spd,
                                'frame_deteksi': frame_diproses
                            }
                            sudah_hitung_up.add(obj_id)
                            counter_up.append(obj_id)
                            print(f"⬆️  ID {obj_id_rapi:>3} ({name:<5}) | {spd:>6.1f} km/h")

            #Tampilkan kecepatan 1 detik setelah terdeteksi
            if obj_id in info_speed:
                df_tampil = frame_diproses - info_speed[obj_id]['frame_deteksi']
                if df_tampil <= DURASI_TAMPIL_FRAME:
                    spd = info_speed[obj_id]['kmh']
                    color_speed = (0,0,255) if spd > 80 else (0,255,255)
                    cv2.putText(frame, f"{spd:.1f} Km/h",
                                (x1, y2+22),
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, color_speed, 2)
                else:
                    # Hapus setelah durasi habis
                    del info_speed[obj_id]

    # Counter
    cv2.putText(frame, f'goingdown: {len(counter_down)}',
                (20,35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'goingup:   {len(counter_up)}',
                (20,70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,255,255), 2)

    # Tulis frame 2x supaya durasi output = durasi asli
    out.write(frame)
    out.write(frame)

    cv2.imshow('RGB - Processing (ESC=stop)', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\nSelesai dalam {time.time()-start_time:.0f} detik")
print(f"\nTotal results:")
print(f"   ⬇️  Down : {len(counter_down)} kendaraan")
print(f"   ⬆️  Up  : {len(counter_up)} kendaraan")