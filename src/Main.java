/**
 * Author: Muhammad Farrel Haidar
 * Project: AI PR & KOL Specialist DSS
 * Date: 2026-09-29
 */

import engine.BadKolCampaignEngine;
import engine.GoodKolCampaignEngine;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("=========================================================");
        System.out.println("  SIMULASI AI PR & KOL SPECIALIST (JAVA 21 ENGINE DEMO)  ");
        System.out.println("=========================================================\n");

        var keyword = "Peluncuran Produk Baru";
        var kolFollowers = 1000; // Base rate: 1000 * 50 = 50.000
        
        // Dummy Data: 3 Positif (+1), 2 Negatif (-1)
        // Seharusnya Total Sentiment = (3 * 1.0) + (2 * -1.0) = 1.0
        // Seharusnya Final Budget = 50000 + (1.0 * 1000) = 51000.0
        var comments = List.of(
                "Produk ini sangat bagus, saya suka!", // +1
                "Keren banget inovasinya", // +1
                "Pengirimannya jelek dan lama", // -1
                "Pelayanannya sangat buruk", // -1

                "Kualitasnya bagus dan awet" // +1
        );

        System.out.println("[ DATA INPUT ]");
        System.out.println("Followers KOL : " + kolFollowers);
        System.out.println("Jumlah Komen  : " + comments.size());
        System.out.println("Komentar Akhir: \"" + comments.get(comments.size() - 1) + "\"\n");

        System.out.println("---------------------------------------------------------");

        // 1. Uji Coba Bad Engine (Terdapat Scope Leak Bug)
        var badEngine = new BadKolCampaignEngine();
        badEngine.processCampaign(keyword, comments, kolFollowers);
        // HASIL BAD ENGINE:
        // Karena variabel di dalam loop bocor, budget akan dihitung menggunakan skor komentar terakhir (-1.0)
        // 50000 + (-1.0 * 1000) = 49000.0 (SALAH BESAR, kerugian secara logic bisnis!)

        System.out.println("---------------------------------------------------------");

        // 2. Uji Coba Good Engine (Single Responsibility Method)
        var goodEngine = new GoodKolCampaignEngine();
        goodEngine.processCampaign(keyword, comments, kolFollowers);
        // HASIL GOOD ENGINE:
        // Variabel terisolasi di method-nya masing-masing. Total sentiment = 1.0
        // 50000 + (1.0 * 1000) = 51000.0 (BENAR)
        
        System.out.println("=========================================================");
    }
}
