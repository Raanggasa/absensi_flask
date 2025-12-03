import os
import cv2
import pandas as pd
import datetime
import time
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace

# Inisialisasi Aplikasi Flask
app = Flask(__name__)

# --- KONFIGURASI DAN SETUP ---

DB_PATH = "dataset"
FILE_ABSENSI = "absensi.csv"

# 1. Hapus file XML lokal yang rusak jika ada (Auto-Fix Error Anda)
local_cascade_file = "haarcascade_frontalface_default.xml"
if os.path.exists(local_cascade_file):
    try:
        os.remove(local_cascade_file)
        print("[INFO] Menghapus file XML lokal yang rusak/bermasalah.")
    except:
        pass

# 2. Pastikan folder dataset ada
if not os.path.exists(DB_PATH):
    os.makedirs(DB_PATH)

# 3. Pastikan file CSV ada
if not os.path.exists(FILE_ABSENSI):
    df = pd.DataFrame(columns=["Nama", "Waktu", "Tanggal"])
    df.to_csv(FILE_ABSENSI, index=False)

# 4. LOAD HAAR CASCADE (METODE AMAN - MENGGUNAKAN BAWAAN LIBRARY)
# Kita ambil path langsung dari library cv2 yang terinstall
system_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

if not os.path.exists(system_cascade_path):
    # Fallback darurat jika path sistem tidak ketemu
    print("[WARNING] Path sistem tidak ditemukan, mencoba load default...")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
else:
    print(f"[INFO] Menggunakan model wajah dari sistem: {system_cascade_path}")
    face_cascade = cv2.CascadeClassifier(system_cascade_path)

# Cek apakah berhasil di-load
if face_cascade.empty():
    print("\n" + "="*50)
    print("CRITICAL ERROR: Masalah pada Instalasi OpenCV.")
    print("Solusi: Buka terminal dan jalankan: pip install --upgrade opencv-python")
    print("="*50 + "\n")
    # Jangan exit, biarkan berjalan tapi tanpa deteksi kotak hijau (Deepface tetap jalan)
else:
    print("[SUCCESS] Model deteksi wajah berhasil dimuat!")

# --- VARIABEL GLOBAL ---
camera = cv2.VideoCapture(0)  
latest_frame = None           
last_recognition_result = "Menunggu..."
last_recognition_color = "secondary"

# --- FUNGSI UTAMA ---

def catat_absensi(nama):
    try:
        if os.path.exists(FILE_ABSENSI):
            df = pd.read_csv(FILE_ABSENSI)
        else:
            df = pd.DataFrame(columns=["Nama", "Waktu", "Tanggal"])

        now = datetime.datetime.now()
        tgl_hari_ini = now.strftime("%Y-%m-%d")
        waktu_sekarang = now.strftime("%H:%M:%S")
        
        cek = df[(df['Nama'] == nama) & (df['Tanggal'] == tgl_hari_ini)]
        
        if cek.empty:
            new_data = pd.DataFrame({"Nama": [nama], "Waktu": [waktu_sekarang], "Tanggal": [tgl_hari_ini]})
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(FILE_ABSENSI, index=False)
            return True, f"Hadir: {nama}"
        else:
            return False, f"Sudah Absen: {nama}"
    except Exception as e:
        return False, "Error System"

def gen_frames():
    global latest_frame, last_recognition_result, last_recognition_color
    frame_count = 0
    process_interval = 30 
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        latest_frame = frame.copy() 
        if frame is None or frame.shape[0] == 0: continue

        # 1. Deteksi Wajah Ringan (Haar)
        if not face_cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                # Scale Factor 1.3 dan minNeighbors 5 adalah setting standar yang stabil
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            except Exception as e:
                pass # Skip error deteksi agar stream tidak putus
        
        # 2. Pengenalan Wajah Akurat (DeepFace)
        frame_count += 1
        if frame_count % process_interval == 0:
            try:
                cv2.imwrite("temp_frame.jpg", frame)
                dfs = DeepFace.find(img_path="temp_frame.jpg", db_path=DB_PATH, 
                                  model_name='VGG-Face', silent=True, enforce_detection=False)
                
                if len(dfs) > 0 and not dfs[0].empty:
                    matched = dfs[0].iloc[0]['identity']
                    nama_user = os.path.basename(os.path.dirname(matched))
                    status, msg = catat_absensi(nama_user)
                    last_recognition_result = msg
                    last_recognition_color = "success" if status else "warning"
                else:
                    last_recognition_result = "Wajah Tidak Dikenal"
                    last_recognition_color = "danger"
            except: pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_absensi')
def status_absensi():
    return jsonify({"message": last_recognition_result, "color": last_recognition_color})

@app.route('/register_face', methods=['POST'])
def register_face():
    nama = request.form.get('nama')
    if not nama: return jsonify({"status": "error", "message": "Nama wajib diisi!"})
    
    safe_nama = "".join([c for c in nama if c.isalpha() or c.isdigit() or c==' ']).strip()
    user_folder = os.path.join(DB_PATH, safe_nama)
    if not os.path.exists(user_folder): os.makedirs(user_folder)
        
    global latest_frame
    count = 0
    max_samples = 20
    timeout = 0
    
    # Gunakan classifier yang aman
    global face_cascade

    try:
        while count < max_samples:
            if latest_frame is None:
                time.sleep(0.1)
                timeout += 1
                if timeout > 50: return jsonify({"status": "error", "message": "Kamera error."})
                continue
            
            current_frame = latest_frame.copy()
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # PROTEKSI ERROR DISINI
            faces = []
            if not face_cascade.empty():
                try:
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                except:
                    # Jika deteksi gagal, kita anggap tidak ada wajah tapi JANGAN CRASH
                    faces = [] 

            if len(faces) > 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{user_folder}/{safe_nama}_{count}_{timestamp}.jpg"
                cv2.imwrite(filename, current_frame)
                count += 1
                time.sleep(0.15)
            else:
                time.sleep(0.1)
        
        # Hapus cache DeepFace
        pkl_path = os.path.join(DB_PATH, "representations_vgg_face.pkl")
        if os.path.exists(pkl_path): os.remove(pkl_path)
            
        return jsonify({"status": "success", "message": f"Berhasil registrasi {safe_nama}!"})
    except Exception as e:
        print(f"ERROR REGISTRASI: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/get_logs')
def get_logs():
    if os.path.exists(FILE_ABSENSI):
        try:
            df = pd.read_csv(FILE_ABSENSI)
            return df.tail(10).to_json(orient="records")
        except: return jsonify([])
    return jsonify([])

@app.route('/delete_all_logs', methods=['POST'])
def delete_all_logs():
    try:
        if os.path.exists(FILE_ABSENSI):
            # Timpa file CSV dengan header saja (kosongkan isinya)
            df = pd.DataFrame(columns=["Nama", "Waktu", "Tanggal"])
            df.to_csv(FILE_ABSENSI, index=False)
            return jsonify({"status": "success", "message": "Semua data berhasil dihapus"})
        return jsonify({"status": "error", "message": "File tidak ditemukan"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)