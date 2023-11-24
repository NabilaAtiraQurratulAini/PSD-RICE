import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

st.title("""DATA RICE CAMMEO AND OSMANCIK""")
st.write("210411100066 - Nabila Atira Qurratul Aini - Proyek Sains Data A")

# Mengasumsikan 'st' adalah instansi Streamlit
tabs = st.tabs(["Dataset Description", "Data File", "Preprocessing", "Support Vector Machine", "Inputan Data"])
dataset_description, data_file, preprocessing, svm, inputan = tabs

with dataset_description:
    st.write("### DESKRIPSI DATASET")
    st.write("Penelitian ini difokuskan pada padi bersertifikat di Turki, dengan penekanan khusus pada dua spesies, yaitu Osmancik dan Cammeo. Osmancik dikenal memiliki luas penanaman yang signifikan sejak tahun 1997, sementara Cammeo mulai ditanam sejak tahun 2014. Sebanyak 3810 gambar butiran beras diambil untuk kedua spesies ini, dan seluruh dataset telah diproses tanpa adanya nilai yang hilang. Dari pengolahan tersebut, diperoleh 7 fitur morfologis untuk setiap butir beras, yang mencakup parameter seperti luas, keliling, panjang sumbu utama, panjang sumbu kecil, eksentrisitas, luas cembung, dan ekstensi. Selain itu, terdapat satu fitur kelas yang menyimpan informasi mengenai kategori atau spesies butir beras, seperti 'Osmancik' atau 'Cammeo'. Karakteristik umum spesies Osmancik mencakup penampilan yang lebar, panjang, kaca, dan kusam, sementara spesies Cammeo memiliki ciri serupa dengan penampilan yang lebar dan panjang, serta kecenderungan kaca dan kusam. Dataset ini dirancang untuk keperluan klasifikasi, dengan tujuan utama mengembangkan model yang dapat mengklasifikasikan butir beras ke dalam spesies Osmancik atau Cammeo berdasarkan fitur-fitur morfologis yang telah diekstrak.")
    st.write("### INFORMASI FITUR")
    st.write("- Area atau Daerah : Fitur ini mengukur luas dari objek. Luas dapat dihitung dalam unit piksel atau unit luas lainnya, tergantung pada resolusi data. Luas memberikan informasi tentang ukuran relatif objek. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Perimeter : Perimeter mengukur panjang garis batas objek. Ini diukur sebagai jumlah panjang semua tepi objek. Perimeter bisa memberikan indikasi seberapa kompleks bentuk objek tersebut. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Major Axis Length atau Panjang Sumbu Utama : Sumbu utama adalah sumbu terpanjang dalam elips yang mengelilingi objek. Panjang sumbu ini memberikan gambaran tentang dimensi utama objek dan arah orientasi elips. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Minor Axis Length atau Panjang Sumbu Kecil : Sumbu kecil adalah sumbu terpendek dalam elips yang mengelilingi objek. Ini memberikan informasi tentang dimensi kedua objek dan dapat membantu menggambarkan bentuk elips. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Eccentricity atau Eksentrisitas : Eksentrisitas mengukur sejauh mana elips yang mengelilingi objek mendekati bentuk lingkaran. Nilai eksentrisitas 0 menunjukkan objek yang bentuknya mendekati lingkaran sempurna, sementara nilai mendekati 1 menunjukkan elips yang sangat panjang. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Convex Area atau Daerah Cembung : Luas cembung mengukur luas daerah yang diukur dari cembung (convex hull) objek. Cembung adalah poligon terkecil yang dapat mencakup seluruh objek. Luas ini memberikan informasi tentang sejauh mana objek dapat dianggap 'cembung'. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Extent atau Luas : Extent adalah rasio antara luas objek dan luas kotak terkecil yang dapat mengelilingi objek. Nilai 1 menunjukkan objek yang mengisi kotak dengan sempurna, sementara nilai yang lebih rendah menunjukkan objek yang mungkin memiliki bentuk yang lebih tidak teratur. Data yang terkait dengan fitur ini memiliki tipe data numerik.")
    st.write("- Class atau Kelas : Kelas adalah label kategori atau jenis keanggotaan dari objek. Ini adalah informasi klasifikasi yang menunjukkan keberadaan objek dalam kategori tertentu, seperti 0 : kelas 'Rice Cammeo' atau 1 : kelas 'Rice Osmancik'. Jumlah data untuk Cammeo adalah 1630, dan jumlah data untuk kelas Osmancik adalah 2180.")

    st.write("### SUMBER DATASET UCI")
    st.write("Sumber dataset rice cammeo and osmancik")
    st.write("https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik")

    st.write("### SOURCE CODE APLIKASI DI GOOGLE COLABORATORY")
    st.write("Code dari dataset rice cammeo and osmancik yang diiunputkan ada di google colaboratory di bawah.")
    st.write("https://colab.research.google.com/drive/1Ch4RqIOTLTu2H0yVwvfzs6859N5RsH51?usp=sharing")

    st.write("### SOURCE CODE APLIKASI DI GITHUB")
    st.write("Code dari dataset rice cammeo and osmancik yang diiunputkan ada di GitHub di bawah.")
    st.write("https://github.com/NabilaAtiraQurratulAini/PSD-RICE.git")

with data_file:
    st.write("### DATA RICE CAMMEO & OSMANCIK")
    st.write("Data")
    df = pd.read_csv("https://raw.githubusercontent.com/NabilaAtiraQurratulAini/PsdA/main/Rice_Osmancik_Cammeo_Dataset.csv")
    df

    column_names = df.columns
    st.write("Nama-nama Kolom dalam Data")
    st.write(column_names)

with preprocessing:
    st.write("### NORMALISASI DATA")
    st.write("Normalisasi adalah proses mengubah nilai-nilai dalam dataset ke dalam skala yang seragam tanpa mengubah struktur relatif di antara nilai-nilai tersebut. Hal ini umumnya dilakukan untuk menghindari perbedaan skala yang besar antara fitur-fitur (kolom-kolom) dalam dataset, yang dapat menyebabkan model machine learning menjadi tidak stabil dan mempengaruhi kinerjanya. Proses normalisasi yang digunakan pada data latih dalam kode ini adalah menggunakan StandardScaler. Hasil normalisasi dari data latih disimpan di dalam file 'scaler.pkl' menggunakan modul pickle. Selanjutnya, saat akan melakukan normalisasi data uji, scaler yang telah disimpan tadi di-load kembali menggunakan pickle, dan data uji dinormalisasi menggunakan scaler yang telah di-load. Hasil normalisasi data uji kemudian ditampilkan dalam bentuk DataFrame.")
    st.write("Beberapa metode normalisasi umum termasuk Min-Max Scaling dan Z-core Standardization :")
    st.write("### Min-Max Scaling")
    st.write("Xscaled = X - Xmin / Xmax - Xmin")
    st.markdown("""
    Penjelasan :
    - Menyebabkan nilai-nilai dalam dataset berada dalam rentang [0, 1].
    - Cocok untuk data yang memiliki distribusi seragam.
    """)
    st.write("### Z-score Standardization")
    st.write("Z = X - μ / σ")
    st.markdown("""
    Penjelasan :
    - Menyebabkan nilai-nilai dalam dataset memiliki rata-rata (μ) 0 dan deviasi standar (σ) 1.
    - Cocok untuk data yang memiliki distribusi normal atau mendekati distribusi normal.
    """)

    df['CLASS'].replace('Cammeo', 0,inplace=True)
    df['CLASS'].replace('Osmancik', 1,inplace=True)

    # langkah 2 : split data menjadi fitur (X) dan target (y)
    X = df.drop(columns=['CLASS'])
    y = df['CLASS']

    # langkah 3 : bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # langkah 4 : normalisasi data menggunakan StandardScaler pada data latih
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # langkah 5 : simpan objek scaler ke dalam file "scaler.pkl" menggunakan pickle
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # langkah 6 : load kembali scaler dari file "scaler.pkl" menggunakan pickle
    with open('scaler.pkl', 'rb') as scaler_file:
        scalerr = pickle.load(scaler_file)

    # langkah 7 : normalisasi data uji menggunakan scaler yang telah disimpan
    X_test_scaler = scalerr.transform(X_test)

    # tampilkan hasil normalisasi data uji
    df_test_scaler = pd.DataFrame(X_test_scaler, columns=X.columns)
    st.write(df_test_scaler.head())

with svm:
    st.write("### METODE SUPPORT VECTOR MACHINE")
    st.write("Support Vector Machine adalah algoritma machine learning yang digunakan untuk masalah klasifikasi dan regresi. Bekerja dengan mencari hiperlane terbaik yang memisahkan dua kelas data dalam ruang berdimensi tinggi. Misalkan memiliki dua kelas data, yaitu kelas 0 dan kelas 1. Untuk menentukan sebuah data termasuk dalam kelas 0 atau kelas 1 dilakukan dengan menghitung prediksi berdasarkan posisi data terhadap hiperplane.")
    st.write("### Rumus Prediksi Kelas Data Baru")
    st.write("f(x) = w * x + b")
    st.markdown("""
    Keterangan :
    - f(x) adalah nilai prediksi untuk data input x.
    - w adalah vektor bobot yang telah ditemukan selama pelatihan SVM.
    - x adalah vektor fitur dari data yang ingin di klasifikasikan.
    - b adalah bias yang juga ditemukan selama pelatihan SVM.
    """)
    st.write("Hasil prediksi f(x) digunakan untuk menetukan kelas data sebagai berikut :")
    st.markdown("""
    - Jika f(x) > 0, maka data akan diklasifikasikan ke dalam kelas 1.
    - Jika f(x) < 0, maka data akan diklasifikasikan ke dalam kelas 0.
    """)
    st.write("Dalam implementasi praktis, nilai ambang (threshold) 0 digunakan, yang berarti jika f(x) sama dengan atau lebih besar dari 0, data diklasifikasikan ke dalam kelas 1; jika kurang dari 0, data diklasifikasikan ke dalam kelas 0.")
    
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y_train)

    with open('svm_model.pkl', 'wb') as svm_model_file:
        pickle.dump(svm_model, svm_model_file)

    with open('svm_model.pkl', 'rb') as svm_model_file:
        loaded_svm_model = pickle.load(svm_model_file)

    y_pred_svm = loaded_svm_model.predict(X_test_scaler)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    st.write("### HASIL AKURASI")
    st.write("Akurasi Terbaik Menggunakan Metode SVM :", accuracy_svm)

    svm_results_df = pd.DataFrame({'Actual Label': y_test, 'Prediksi SVM': y_pred_svm})
    svm_results_df

with inputan:
    st.write("### APLIKASI PREDIKSI JENIS BERAS")

    # formulir input untuk nilai-nilai fitur
    AREA = st.number_input("Masukkan nilai area : ")
    PERIMETER = st.number_input("Masukkan nilai perimeter : ")
    MAJORAXIS = st.number_input("Masukkan nilai majoraxis : ")
    MINORAXIS = st.number_input("Masukkan nilai minoraxis : ")
    ECCENTRICITY = st.number_input("Masukkan nilai eccentricity : ")
    CONVEX_AREA = st.number_input("Masukkan nilai convex area : ")
    EXTENT = st.number_input("Masukkan nilai extent : ")

    # tombol untuk membuat prediksi
    if st.button("Prediksi"):
        # membuat array dengan data input
        new_data = np.array([[AREA, PERIMETER, MAJORAXIS, MINORAXIS, ECCENTRICITY, CONVEX_AREA, EXTENT]])

        # normalisasi data input menggunakan scaler yang disimpan
        new_data_scaled = scalerr.transform(new_data)

        # melakukan prediksi menggunakan model SVM
        prediction = loaded_svm_model.predict(new_data_scaled)

        # menampilkan hasil prediksi
        if prediction[0] == 1:
            st.write("Hasil Prediksi : Beras Osmancik")
        else:
            st.write("Hasil Prediksi : Beras Cammeo")
