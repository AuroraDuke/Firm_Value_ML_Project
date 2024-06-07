# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 23:49:58 2024

@author: ilker
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

datas=pd.read_csv('BalanceSheetDataSet.csv')

#şirket şirket yapılacaktı sonra tüm data set ile yapma kararı alındı
datas_column_names = list(datas.columns)
datas = datas.iloc[::-1]
sirket = datas.drop(columns=['Dönem Net Kar/Zararı'])
output = datas['Dönem Net Kar/Zararı']

#isimler gözükmesinde sıkıntı olduğundan dolayı bunu 0,1,2,3 gibi sayılar verdim
new_column_names = [str(i) for i in range(len(datas.columns))]
datas.columns = new_column_names

encoded_data = pd.get_dummies(datas.iloc[:, 0], prefix='A').astype(int)
data_encoded = pd.concat([encoded_data, datas.iloc[:, 1:]], axis=1)

#corr------------------------------------------------------------------------
import seaborn as sns
#plt.rcParams['figure.dpi'] = 300
# Grafik boyutlarını ayarlama (örneğin, 1000x600 piksel)
#plt.figure(figsize=(1000/300, 600/300))
print("Veri seti içindeki değişkenlerin birbiri ile ilişki katsayısı")
corr=np.abs(data_encoded.corr(method='pearson'))
#corr
column_index = 27
# Seçilen sütunun adı
selected_column_name = data_encoded.columns[column_index]
# Seçilen sütunun korelasyonları
column_correlations = corr[selected_column_name]
# Korelasyon eşiği
threshold = 0.5
# Eşik altındaki korelasyonları filtrele
filtered_correlations = column_correlations[abs(column_correlations) >= threshold]
# Filtrelenmiş korelasyonların sütunlarını al
selected_columns = filtered_correlations.index
# Yeni korelasyon matrisini oluştur
filtered_corr = corr.loc[selected_columns, selected_columns]

#Heatmap--------------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(filtered_corr, annot=False, fmt=".2f", linewidths=0.9, center=0)
plt.title(f"Korelasyon Matrisi (Seçilen Sütun ile Korelasyon Eşiği: 0.5)")
plt.show()
new_df = data_encoded[selected_columns].copy()
#draw grafikleştirme 10ar 10ar gösteren code------------------------------------------------------------------------

new_df = new_df.drop('3', axis = 1)

selected_datas_cloumns_name=[]
for i in range(len(selected_columns)):
      selected_datas_cloumns_name.append(  datas_column_names[int(selected_columns[i])] )
      
total_columns = len(new_df.columns)

# Her seferinde kaç sütun göstereceğimiz
num_cols_per_plot = 10

# Toplam kaç adet subplot olacağı
num_plots = total_columns // num_cols_per_plot
if total_columns % num_cols_per_plot != 0:
    num_plots += 1

# Plotları oluşturma
for plot_index in range(num_plots):
    start_index = plot_index * num_cols_per_plot
    end_index = min((plot_index + 1) * num_cols_per_plot, total_columns)
    
    # Subplot boyutlarını ayarla
    plt.figure(figsize=(20, 15))
    #
    for i, col in enumerate(new_df.columns[start_index:end_index]):
        plt.subplot(2, 5, i+1)  # 2 satır, 5 sütunluk bir düzen
        x = new_df[col]
        y = output
        plt.plot(x, y, 'o')
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('prices')

    plt.tight_layout()
    plt.show()
    #hist
    for i, col in enumerate(new_df.columns[start_index:end_index]):
        plt.subplot(2, 5, i+1)  # 2 satır, 5 sütunluk bir düzen
        new_df[col].hist(bins=10, grid=False)
        plt.title(col)
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


for i in range(1, len(new_df) + 10, 10):
    n = min(i + 10, len(new_df))
    new_df.iloc[:, i:n].plot(kind='box', layout=(1,10), sharex=False, sharey=False)
    plt.show()

