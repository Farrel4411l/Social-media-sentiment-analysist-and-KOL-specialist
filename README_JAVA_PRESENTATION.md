<!--
/**
 * Author: Muhammad Farrel Haidar
 * Project: AI PR & KOL Specialist DSS
 * Date: 2026-09-29
 */
-->

# Presentasi OOP: The Benefits of Breaking Programs into Methods

Repositori ini adalah suplemen praktis untuk presentasi perkuliahan *Object-Oriented Programming (Java)*, mendemonstrasikan bahaya dari sebuah *God Method* dan bagaimana memecahnya dapat mencegah *Silent Bug*.

## 🚀 Cara Menjalankan (Compile & Run)

Pastikan Anda telah menginstal **JDK 21** atau yang lebih baru. Buka Terminal/Command Prompt, lalu arahkan ke direktori proyek Anda.

1. **Pindah ke direktori sumber (`src`)**:
   ```bash
   cd src
   ```

2. **Kompilasi (Compile)** file `.java`:
   ```bash
   javac Main.java engine/BadKolCampaignEngine.java engine/GoodKolCampaignEngine.java
   ```

3. **Jalankan (Run)** program:
   ```bash
   java Main
   ```

---

## 🧠 Materi Presentasi Akademik (Cheat Sheet)

Bawakan poin-poin berikut saat presentasi di depan kelas:

### 1. "Silent Bug" Akibat Kegagalan Isolasi Variabel (Scope Leak)
Dalam file `BadKolCampaignEngine.java`, seluruh logika berada dalam satu *God Method*. Developer secara tidak sengaja menggunakan ulang (re-use) variabel `score` (yang seharusnya hanya untuk sentimen lokal tiap komentar) di dalam operasi agregat final di loop yang sama.
- **Bahaya:** *Compiler* Java tidak akan memunculkan error (Syntax OK). Namun, secara logika *budget* bisnis, angka akhirnya salah parah karena hanya menghitung data dari komentar yang terakhir kali dilooping.
- **Solusi Metode (Good Engine):** Dengan memecahnya di `GoodKolCampaignEngine.java`, kita mengisolasi siklus hidup (lifecycle) dari sebuah variabel. Metode `evaluateSingleComment()` hanya hidup sesaat, melempar nilai _return_, lalu memori variabel `score` dihancurkan *(Garbage Collected)*, sehingga mustahil tertimpa atau dipakai ulang tanpa disengaja di perhitungan *budget*.

### 2. Efisiensi Memori & Garbage Collection
Memecah metode berarti siklus hidup variabel *(Variable Lifecycle)* menjadi sangat pendek. Variabel yang dibuat di dalam metode kecil akan segera masuk ke antrean *Garbage Collector* segera setelah metode selesai (kembali). Pada *God Method*, variabel yang tidak lagi digunakan akan tetap hidup di RAM (*Heap/Stack*) selama blok raksasa tersebut masih berjalan.

### 3. Skalabilitas & Cross-Platform (Pure Java POJO)
Struktur *engine* ini dirancang menggunakan *Pure Object* (POJO) tanpa ketergantungan UI *framework* apa pun (seperti Swing, JavaFX, atau Android SDK).
Artinya, fungsi `GoodKolCampaignEngine.processCampaign()` ini 100% *reusable*. Anda bisa menyuntikkannya ke *Backend* Spring Boot, mendaur ulangnya di Android Studio (Mobile), atau memanggilnya di aplikasi Desktop GUI dengan hasil komputasi yang terjamin keamanannya.
