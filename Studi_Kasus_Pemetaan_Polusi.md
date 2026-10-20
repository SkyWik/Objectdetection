# Studi Kasus: Deteksi Objek Sampah untuk Pemetaan Polusi Lingkungan

## Latar Belakang
Pemetaan polusi lingkungan, seperti penumpukan sampah di area publik (contoh lokasi: Desa Pasar Baru), membutuhkan pemantauan spasial yang akurat. Identifikasi sebaran limbah secara manual seringkali memakan waktu dan tenaga. Dengan memanfaatkan teknologi Computer Vision dan arsitektur Deep Learning (seperti Convolutional Neural Networks atau model berbasis YOLO), kita dapat mengotomatisasi deteksi objek sampah dari citra kamera. Hasil deteksi koordinat ini kemudian dapat diekspor untuk divisualisasikan pada software pemetaan spasial seperti QGIS.

## Tujuan Proyek
Membangun dan melatih prototipe model Deep Learning berbasis CNN menggunakan PyTorch untuk mendeteksi dan mengklasifikasikan jenis sampah (misalnya: Plastik, Organik, Kertas) dari citra lingkungan.

## Struktur Direktori
- `model.py` : Berisi definisi arsitektur neural network (CNN).
- `train.py` : Berisi skrip untuk melakukan proses training model.
- `requirements.txt` : Daftar library Python yang dibutuhkan.

## Langkah Implementasi Selanjutnya
Setelah model dilatih dan memiliki akurasi yang baik, model ini dapat diekspor ke format ONNX dan dihubungkan ke backend (misalnya menggunakan Express.js atau Golang) untuk menerima gambar dari perangkat IoT dan mengembalikan hasil deteksi secara real-time.
