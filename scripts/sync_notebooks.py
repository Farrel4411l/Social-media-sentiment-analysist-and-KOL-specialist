# Author: Muhammad Farrel Haidar
# Project: AI PR & KOL Specialist DSS
# Date: 2026-10-15

"""
Sinkronisasi Notebooks 02 dan 05 dengan data validasi inter-annotator 300 sampel (kappa = 0.8662)
dan audit konsistensi heuristik leksikal (89.39%).
"""

import json
import os

def sync_notebooks():
    # 1. Update Notebook 02
    nb2_path = "notebooks/02_preprocessing.ipynb"
    if os.path.exists(nb2_path):
        with open(nb2_path, "r", encoding="utf-8") as f:
            nb2 = json.load(f)

        nb2["cells"][0]["source"] = [
            s.replace("Cohen's Kappa = 0.8939 (Almost Perfect Agreement, 500 Samples)", 
                      "Cohen's Kappa = 0.8662 (Almost Perfect Agreement, 300 Samples Dual-Annotators) & Heuristic Audit = 89.39%")
            for s in nb2["cells"][0]["source"]
        ]
        nb2["cells"][0]["source"] = [
            s.replace("500 sampel terstratifikasi menghasilkan $\\kappa = 0.8939$", 
                      "300 sampel terstratifikasi dual-annotator menghasilkan $\\kappa = 0.8662$ dan audit heuristik 89.39%")
            for s in nb2["cells"][0]["source"]
        ]

        nb2["cells"][10]["source"] = [
            "### 6. Validasi Manual Inter-Annotator (300 Sampel) & Audit Konsistensi Heuristik\n",
            "Sesuai standar jurnal Scopus Q2/Q3, reliabilitas ground-truth diverifikasi secara formal menggunakan koefisien kesepakatan inter-annotator Cohen's Kappa ($\\kappa$) antara 2 penilai domain independen (PR Specialist vs QA/Operations Specialist) pada 300 sampel terstratifikasi, serta audit konsistensi heuristik aturan leksikal rating (89.39%)."
        ]

        nb2["cells"][11]["source"] = [
            "import json\n",
            "import os\n",
            "import pandas as pd\n",
            "from sklearn.metrics import cohen_kappa_score, accuracy_score\n",
            "\n",
            "val_file = \"../data/manual_validation_300_dual_annotators.csv\" if os.path.exists(\"../data/manual_validation_300_dual_annotators.csv\") else \"data/manual_validation_300_dual_annotators.csv\"\n",
            "val_df = pd.read_csv(val_file)\n",
            "print(f\"Dual-Annotator Validation Dataset Loaded: {len(val_df)} rows\")\n",
            "\n",
            "# Hitung koefisien Cohen's Kappa antara 2 penilai independen\n",
            "kappa = cohen_kappa_score(val_df[\"annotator_1_pr_expert\"], val_df[\"annotator_2_qa_expert\"])\n",
            "acc = accuracy_score(val_df[\"annotator_1_pr_expert\"], val_df[\"annotator_2_qa_expert\"])\n",
            "print(f\"Human Inter-Annotator Agreement (Cohen's Kappa): {kappa:.4f}\")\n",
            "print(f\"Observed Agreement Percentage: {acc*100:.2f}%\")\n",
            "\n",
            "# Tampilkan metadata IAA dan Heuristic Audit JSON\n",
            "iaa_json = \"../paper_assets/cohen_kappa_inter_annotator_agreement.json\" if os.path.exists(\"../paper_assets/cohen_kappa_inter_annotator_agreement.json\") else \"paper_assets/cohen_kappa_inter_annotator_agreement.json\"\n",
            "audit_json = \"../paper_assets/label_consistency_audit.json\" if os.path.exists(\"../paper_assets/label_consistency_audit.json\") else \"paper_assets/label_consistency_audit.json\"\n",
            "\n",
            "if os.path.exists(iaa_json):\n",
            "    with open(iaa_json, \"r\", encoding=\"utf-8\") as f:\n",
            "        print(\"\\n--- INTER-ANNOTATOR AGREEMENT METADATA ---\")\n",
            "        print(json.dumps(json.load(f), indent=2))\n",
            "\n",
            "if os.path.exists(audit_json):\n",
            "    with open(audit_json, \"r\", encoding=\"utf-8\") as f:\n",
            "        print(\"\\n--- HEURISTIC CONSISTENCY AUDIT METADATA ---\")\n",
            "        print(json.dumps(json.load(f), indent=2))\n"
        ]

        with open(nb2_path, "w", encoding="utf-8") as f:
            json.dump(nb2, f, indent=1)
        print("Updated notebook 02!")

    # 2. Update Notebook 05
    nb5_path = "notebooks/05_comparative_analysis.ipynb"
    if os.path.exists(nb5_path):
        with open(nb5_path, "r", encoding="utf-8") as f:
            nb5 = json.load(f)

        nb5["cells"][0]["source"] = [
            s.replace("($\\kappa = 0.8939$)", "($\\kappa = 0.8662$ IAA, 89.39% Heuristic Audit)")
            for s in nb5["cells"][0]["source"]
        ]

        nb5["cells"][9]["source"] = [
            "iaa_path = \"../paper_assets/cohen_kappa_inter_annotator_agreement.json\" if os.path.exists(\"../paper_assets/cohen_kappa_inter_annotator_agreement.json\") else \"paper_assets/cohen_kappa_inter_annotator_agreement.json\"\n",
            "audit_path = \"../paper_assets/label_consistency_audit.json\" if os.path.exists(\"../paper_assets/label_consistency_audit.json\") else \"paper_assets/label_consistency_audit.json\"\n",
            "\n",
            "if os.path.exists(iaa_path):\n",
            "    with open(iaa_path, \"r\", encoding=\"utf-8\") as f:\n",
            "        print(\"=== HUMAN INTER-ANNOTATOR AGREEMENT (IAA) ===\")\n",
            "        print(json.dumps(json.load(f), indent=2))\n",
            "\n",
            "if os.path.exists(audit_path):\n",
            "    with open(audit_path, \"r\", encoding=\"utf-8\") as f:\n",
            "        print(\"\\n=== HEURISTIC CONSISTENCY AUDIT ===\")\n",
            "        print(json.dumps(json.load(f), indent=2))\n"
        ]

        with open(nb5_path, "w", encoding="utf-8") as f:
            json.dump(nb5, f, indent=1)
        print("Updated notebook 05!")

if __name__ == "__main__":
    sync_notebooks()
