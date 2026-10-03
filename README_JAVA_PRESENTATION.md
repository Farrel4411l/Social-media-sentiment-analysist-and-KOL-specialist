<!--
/**
 * Author: Muhammad Farrel Haidar
 * Project: AI PR & KOL Specialist DSS
 * Date: 2026-09-30
 */
-->

# 📖 Dokumentasi Lengkap: "The Benefits of Breaking Programs into Methods"

Selamat datang di repositori suplemen mata kuliah **Object-Oriented Programming (Java)**. Proyek ini dibangun dengan standar arsitektur industri (*Clean Code*) untuk mendemonstrasikan secara empiris mengapa memecah sebuah *God Method* (metode raksasa) menjadi metode-metode kecil (pendekatan *Single Responsibility*) adalah keharusan, bukan sekadar gaya penulisan.

## 🎯 Objektif Proyek
Proyek ini mengambil studi kasus nyata dari sistem **AI PR & KOL Specialist DSS**. Seringkali, saat memproses data sosial media berjumlah ribuan, *developer* pemula menempatkan logika ekstraksi, *looping* sentimen, dan kalkulasi *budget* di dalam satu metode besar. 
Proyek ini secara sengaja mendemonstrasikan **"Silent Bug"**—sebuah kesalahan di mana kode berhasil di-compile (Syntax OK), tidak menghasilkan Exception (Runtime OK), namun secara diam-diam menghasilkan perhitungan logika bisnis yang fatal akibat kebocoran cakupan variabel (*Scope Leak*).

## 📁 Struktur Direktori & Arsitektur
Proyek ini dibangun menggunakan **Java 21** dengan memanfaatkan fitur modern seperti `var` (*Type Inference*) dan *Switch Expressions*. 

```text
src/
├── Main.java                        # Driver class, tempat eksekusi simulasi
└── engine/
    ├── BadKolCampaignEngine.java    # Demonstrasi Anti-Pattern (God Method & Scope Leak)
    └── GoodKolCampaignEngine.java   # Demonstrasi Best Practice (Method Separation)
```

### 1. `BadKolCampaignEngine.java` (The Anti-Pattern)
- **Karakteristik:** Memiliki satu metode raksasa bernama `processCampaign()`.
- **Anatomi Bug:** Di dalam blok *looping* `for`, terdapat variabel `score`. Karena keseluruhan perhitungan final *budget* dilakukan di blok yang sama, developer secara tidak sengaja menggunakan ulang (*re-use*) variabel `score` tersebut di akhir fungsi. 
- **Akibat:** Nilai agregat total sentimen terabaikan. Sistem menghitung budget berdasarkan sentimen komentar yang *paling terakhir* di-loop. Jika komentar terakhir negatif, budget anjlok meskipun 99% komentar lainnya positif.

### 2. `GoodKolCampaignEngine.java` (The Best Practice)
- **Karakteristik:** Menerapkan *Single Responsibility Principle*. Metode raksasa dipecah menjadi tiga fungsionalitas spesifik:
  1. `calculateAggregateSentiment(List<String> comments)` -> Khusus agregasi.
  2. `evaluateSingleComment(String comment)` -> Khusus evaluasi skor individual menggunakan fitur modern Java *Switch Expression*.
  3. `calculateKolBudget(int kolFollowers, double finalSentimentScore)` -> Khusus perhitungan akhir.
- **Dampak Penyelamatan:** Dengan memecah metode, *Scope* memori variabel `score` secara paksa terisolasi di dalam `evaluateSingleComment()`. Tidak mungkin bagi metode `calculateKolBudget()` untuk mengakses variabel tersebut secara keliru, memaksa developer untuk mem-passing data secara eksplisit (sebagai `finalSentimentScore`).

## 🛠️ Cara Eksekusi (Troubleshooting JVM)

Kompilasi proyek ini mewajibkan **JDK 21/23** (karena penggunaan `var` dan *Switch Expression*).

### Melalui IDE (Direkomendasikan)
Untuk kemudahan presentasi tanpa kendala versi CLI:
1. Buka folder `src` menggunakan **IntelliJ IDEA** atau **Eclipse**.
2. Pastikan *Project Structure > Project SDK* disetel ke Java 21 atau 23.
3. Buka `Main.java` dan tekan tombol **Run (▶)**.

### Melalui CLI / Terminal
Jika Anda menjalankan via CLI, pastikan `javac -version` dan `java -version` keduanya menunjuk ke versi 21+. 
```bash
# Masuk ke direktori
cd src

# Kompilasi
javac Main.java engine/BadKolCampaignEngine.java engine/GoodKolCampaignEngine.java

# Jalankan
java Main
```

*(Catatan: Jika terjadi `UnsupportedClassVersionError` versi 67 ke 52, itu berarti Terminal Anda me-routing perintah `java` ke JRE 8. Gunakan IDE untuk menghindari isu *Path Environment* Windows saat presentasi).*
