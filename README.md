# 📸 Sistem Absensi AI: Face Recognition Attendance

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask&logoColor=white)
![DeepFace](https://img.shields.io/badge/AI-DeepFace-orange?style=for-the-badge&logo=meta&logoColor=white)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-green?style=for-the-badge&logo=opencv&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Frontend-Bootstrap%205-purple?style=for-the-badge&logo=bootstrap&logoColor=white)

**Sistem Absensi AI** adalah aplikasi web modern yang dirancang untuk mencatat kehadiran secara otomatis menggunakan teknologi pengenalan wajah (*Face Recognition*) secara real-time.

Proyek ini menggabungkan kekuatan **DeepFace** (VGG-Face Model) untuk akurasi pengenalan wajah dan **Flask** sebagai backend server, dilengkapi dengan notifikasi suara dan pelaporan data otomatis.

---

## 🌟 Fitur Utama

Aplikasi ini dilengkapi dengan berbagai fitur cerdas untuk memudahkan manajemen absensi:

* **📷 Real-time Face Recognition**
    Mendeteksi dan mengenali wajah secara langsung melalui webcam menggunakan algoritma Deep Learning.
* **👤 Registrasi Wajah Otomatis**
    Fitur pendaftaran pengguna baru yang secara otomatis mengambil 30 sampel foto wajah untuk melatih model.
* **🗣️ Interaksi Suara (Voice Feedback)**
    Sistem menyapa pengguna (Text-to-Speech) saat berhasil absen dan memberi peringatan jika wajah tidak dikenali.
* **⏱️ Smart Cooldown & Anti-Spam**
    Logika cerdas untuk mencegah pencatatan ganda dalam waktu singkat (default: 60 detik).
* **📄 Laporan & Ekspor Data**
    Melihat log kehadiran terkini dan fitur **Download PDF** untuk rekap data absensi.

---

## 🛠️ Teknologi yang Digunakan

Sistem ini dibangun menggunakan teknologi open-source berikut:

* **Backend:** Python, Flask
* **Computer Vision:** OpenCV (Haar Cascade untuk deteksi), DeepFace (VGG-Face untuk pengenalan)
* **Data Management:** Pandas (CSV Handling)
* **Frontend:** HTML5, Bootstrap 5, JavaScript (jQuery & SweetAlert2)
* **Reporting:** jsPDF (Client-side PDF generation)

---

## 📂 Struktur Folder

Pastikan susunan folder proyek Anda seperti berikut agar aplikasi berjalan lancar:

```text
/
├── app.py                  # Server utama (Flask & Logic AI)
├── requirements.txt        # Daftar library python
├── absensi.csv             # Database log (Dibuat otomatis)
├── templates/              # [PENTING] Folder untuk HTML
│   └── index.html          # Antarmuka aplikasi
├── dataset/                # Folder foto wajah user (Dibuat otomatis)
└── ds_model_vggface...pkl  # Cache model DeepFace
