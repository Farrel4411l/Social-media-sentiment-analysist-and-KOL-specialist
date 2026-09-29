/**
 * Author: Muhammad Farrel Haidar
 * Project: AI PR & KOL Specialist DSS
 * Date: 2026-09-29
 */
package engine;

import java.util.List;

public class GoodKolCampaignEngine {

    /**
     * Orchestrator method yang mendelegasikan tugas ke method-method private yang lebih kecil (Single Responsibility).
     * Terbebas dari scope leak bug karena variabel benar-benar terisolasi di method masing-masing.
     */
    public void processCampaign(String keyword, List<String> comments, int kolFollowers) {
        System.out.println("Memproses campaign (GOOD ENGINE) untuk keyword: " + keyword);

        var totalSentiment = calculateAggregateSentiment(comments);
        var finalBudget = calculateKolBudget(kolFollowers, totalSentiment);

        System.out.println("Total Sentiment Score : " + totalSentiment);
        System.out.println("Final Approved Budget : Rp " + finalBudget + "\n");
    }

    private double calculateAggregateSentiment(List<String> comments) {
        var total = 0.0;
        for (var comment : comments) {
            total += evaluateSingleComment(comment);
        }
        return total;
    }

    private double evaluateSingleComment(String comment) {
        // Menggunakan Java 21 Switch Expressions & var untuk kode yang lebih bersih
        var lowerComment = comment.toLowerCase();
        
        // Ekstraksi keyword utama untuk keperluan demonstrasi Switch Expression
        var keyword = lowerComment.contains("bagus") || lowerComment.contains("keren") ? "positif" :
                      lowerComment.contains("jelek") || lowerComment.contains("buruk") ? "negatif" : "netral";

        return switch (keyword) {
            case "positif" -> 1.0;
            case "negatif" -> -1.0;
            default -> 0.0;
        };
    }

    private double calculateKolBudget(int kolFollowers, double finalSentimentScore) {
        // Perhitungan terisolasi dengan baik, 'score' dari sentimen ditarik masuk sebagai parameter
        return (kolFollowers * 50.0) + (finalSentimentScore * 1000.0);
    }
}
