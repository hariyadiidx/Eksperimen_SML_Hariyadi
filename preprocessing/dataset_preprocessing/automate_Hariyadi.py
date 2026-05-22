import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def run_preprocessing():
    print("Memulai Automasi Preprocessing (CI/CD Pipeline)...")

    # 1. Setup Path (Gunakan path relatif agar berfungsi di GitHub Actions)
    raw_path = "dataset_raw/credit_raw.csv"
    processed_dir = "dataset_preprocessing"
    processed_path = f"{processed_dir}/credit_processed.csv"

    # Validasi keberadaan file
    if not os.path.exists(raw_path):
        print(f"❌ Error: File {raw_path} tidak ditemukan!")
        return

    df = pd.read_csv(raw_path)

    # 2. Menangani Missing Values
    modus_credit = df['Credit_History'].mode()[0]
    df['Credit_History'] = df['Credit_History'].fillna(modus_credit)

    # 3. Penanganan Outlier (Capping pada Income)
    batas_atas_income = df['Income'].quantile(0.99)
    df['Income'] = np.where(df['Income'] > batas_atas_income, batas_atas_income, df['Income'])

    # 4. Encoding Data Kategorikal
    le = LabelEncoder()
    df['Credit_History'] = le.fit_transform(df['Credit_History'])

    # 5. Standarisasi Fitur Numerik
    scaler = StandardScaler()
    kolom_numerik = ['Age', 'Income', 'Loan_Amount']
    df[kolom_numerik] = scaler.fit_transform(df[kolom_numerik])

    # 6. Simpan Data Bersih
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"✅ Automasi Preprocessing Selesai! Data siap dilatih dan disimpan di {processed_path}")

if __name__ == "__main__":
    run_preprocessing()
