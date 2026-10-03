/**
 * Author: Muhammad Farrel Haidar
 * Project: AI PR & KOL Specialist DSS
 * Date: 2026-09-29
 */
package engine;

import java.util.List;

public class BadKolCampaignEngine {

    /**
     * Sebuah "God Method" yang melakukan semuanya di satu tempat.
     * Secara sengaja mengandung "Scope Leak / Variable Re-use Bug" yang tidak terdeteksi oleh compiler.
     */
    public void processCampaign(String keyword, List<String> comments, int kolFollowers) {
        System.out.println("Memproses campaign (BAD ENGINE) untuk keyword: " + keyword);

        var totalSentiment = 0.0;
        var finalBudget = 0.0;

        // 'for' memulai perulangan (loop). Ia akan memproses teks dari 'comments' satu per satu secara berurutan.
        for (var comment : comments) {
            
            // Deklarasi variabel 'score'. Karena letaknya di DALAM blok loop, 
            // variabel ini akan selalu "dilahirkan ulang" dengan nilai 0.0 di setiap putaran loop.
            var score = 0.0; 

            // Cek teks: jika kalimat mengandung kata "bagus" atau "keren", beri poin positif (+1.0).
            if (comment.toLowerCase().contains("bagus") || comment.toLowerCase().contains("keren")) {
                score = 1.0;
            // Jika kalimat mengandung kata "jelek" atau "buruk", beri poin negatif (-1.0).
            } else if (comment.toLowerCase().contains("jelek") || comment.toLowerCase().contains("buruk")) {
                score = -1.0;
            } else {
                score = 0.0;
            }

            // Variabel 'totalSentiment' (yang berada di LUAR loop) terus diakumulasikan.
            // Sampai di sini logikanya sudah benar, nilai total gabungan 5 komentar berhasil direkam (skor akhirnya = 1.0).
            totalSentiment += score;

            // ⚠️ KESALAHAN FATAL (LOGIC BUG & SCOPE LEAK) BERADA DI BARIS BAWAH INI:
            // 1. Rumus budget ini posisinya berada DI DALAM blok loop '{ ... }'.
            // 2. Akibatnya, variabel 'finalBudget' secara buta dihitung dan DITIMPA ulang secara terus-menerus di setiap putaran.
            // 3. Kesalahan tergawat: developer memakai variabel 'score' (skor satuan 1 komentar) BUKAN 'totalSentiment' (skor keseluruhan 5 komentar)
            //    untuk dimasukkan ke dalam rumus uang ini.
            // 4. Ketika putaran berakhir di komentar ke-5 (yg bersentimen -1), nilai 'finalBudget' dari komentar 1 sampai 4 HANGUS 
            //    (karena ditimpa oleh hasil hitungan baris ini di putaran terakhir).
            finalBudget = (kolFollowers * 50.0) + (score * 1000.0);
        }

        System.out.println("Total Sentiment Score : " + totalSentiment);
        System.out.println("Final Approved Budget : Rp " + finalBudget + "\n");
    }
}
