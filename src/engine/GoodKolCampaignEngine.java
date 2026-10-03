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

        // LANGKAH 1: Kita panggil metode khusus yang TUGASNYA MURNI HANYA MENGHITUNG SENTIMEN.
        // Kita simpan hasilnya ke variabel 'totalSentiment'. Loop terjadi jauh di dalam perut metode ini.
        var totalSentiment = calculateAggregateSentiment(comments);
        
        // LANGKAH 2: Kita panggil metode lain yang TUGASNYA MURNI HANYA MENGHITUNG UANG (BUDGET).
        // Kita beri makan (parameter) 'totalSentiment' yang sudah direkapitulasi secara tepat di langkah 1.
        // ✅ SOLUSI BUG: Di titik ini, kita sedang berada di LUAR LOOPING dan mustahil bagi variabel 
        // skor satuan individual untuk 'bocor' atau menyelinap ke sini karena variabelnya sudah diisolasi di metode lain.
        var finalBudget = calculateKolBudget(kolFollowers, totalSentiment);

        System.out.println("Total Sentiment Score : " + totalSentiment);
        System.out.println("Final Approved Budget : Rp " + finalBudget + "\n");
    }

    private double calculateAggregateSentiment(List<String> comments) {
        var total = 0.0;
        
        // Loop berjalan secara berurutan.
        for (var comment : comments) {
            // Loop ini "bodoh" (tapi ini hal yang bagus!). Ia tidak tahu cara membaca teks.
            // Ia mendelegasikan (melemparkan) tugas deteksi teks ke metode 'evaluateSingleComment'.
            // Ia hanya fokus mengakumulasi (+=) nilai (0.0 + 1.0 + (-1.0) dsb) yang dikembalikan metode tersebut.
            total += evaluateSingleComment(comment);
        }
        return total;
    }

    private double evaluateSingleComment(String comment) {
        // Metode ini sangat terisolasi. Tugasnya 100% membedah teks, ia tidak tahu-menahu urusan uang/budget.
        // Karena metode kecil dan berumur pendek, ketika nilainya di-return ke atas, 
        // seluruh variabel sementaranya (lowerComment, keyword) akan dianggap sampah dan langsung disapu habis (Garbage Collection) dari RAM.
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
        // Perhitungan aman terkendali. Variabel 'finalSentimentScore' dijamin berisi 
        // nilai sentimen keseluruhan (misal 1.0) yang dioper secara sengaja lewat parameter.
        return (kolFollowers * 50.0) + (finalSentimentScore * 1000.0);
    }
}
