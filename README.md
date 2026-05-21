# YOLOv8 Vehicle Speed Detection

## Deskripsi
Sistem deteksi kendaraan dan estimasi kecepatan menggunakan YOLOv8 + Object Tracking.

## Fitur
- Deteksi kendaraan (mobil dan truk)
- Tracking kendaraan menggunakan BOTSORT
- Perhitungan deteksi kecepatan kendaraan (km/h)
- Deteksi arah kendaraan (naik/turun)
- Counter jumlah kendaraan

## Dataset
Video cctv di ruas jalan tol Semarang ABC, JPO Mrican KM 432

## Library
- OpenCV
- YOLOv8
- Ultralytics
- NumPy
- PyTorch

## Cara Menjalankan

1. Buat virtual environment dan aktifkan :

- python -m venv venv
- venv\Scripts\activate

2. Install dependency

pip install -r requirements.txt

3. Siapkan File Video 

4. Tentukan posisi garis deteksi dengan klik posisi yang diinginkan, kemudian salin outputnya pada main.py & python debug_line.py

python check_line.py

5. Validasi garis deteksi, apakah garis sesuai & pastikan kendaraan benar masuk area L1 & L2

python debug_line.py

6. Jalankan sistem

python main.py

## Struktur Folder

YOLOv8-Vehicle-Speed-Detection/
│
├── videos/
│   └── vehicle_detection1.mp4
│
├── output/
│   └── output_results1.mp4
│
├── main.py
├── debug_line.py
├── check_line.py
├── requirements.txt
├── README.md
└── yolov8s.pt