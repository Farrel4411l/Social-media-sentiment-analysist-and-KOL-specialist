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

        for (var comment : comments) {
            var score = 0.0; // Variabel lokal untuk sentimen tiap komentar

            // Logika sangat sederhana untuk evaluasi sentimen
            if (comment.toLowerCase().contains("bagus") || comment.toLowerCase().contains("keren")) {
                score = 1.0;
            } else if (comment.toLowerCase().contains("jelek") || comment.toLowerCase().contains("buruk")) {
                score = -1.0;
            } else {
                score = 0.0;
            }

            totalSentiment += score;

            // JEBAKAN: Developer secara keliru menggunakan variabel 'score' yang ditujukan untuk 
            // evaluasi sentimen tunggal sebagai pengali budget di dalam loop yang sama!
            // Harusnya menggunakan 'totalSentiment' di luar loop. 
            // Akibatnya, finalBudget hanya akan bergantung pada komentar TERAKHIR di list.
            finalBudget = (kolFollowers * 50.0) + (score * 1000.0);
        }

        System.out.println("Total Sentiment Score : " + totalSentiment);
        System.out.println("Final Approved Budget : Rp " + finalBudget + "\n");
    }
}
