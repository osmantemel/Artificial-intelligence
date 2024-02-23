#Plots kısmında çizim olur

from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)
print(iris.target_names)

print(iris.target)
print(iris.data)

# X -> Giriş/Bağımsız , Y -> Çıkış/Bağımlı

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("Eğitim veriseti boyutu :", len(X_train))
print("Test veriseti boyutu :", len(X_test))

# Karar ağacı

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

# Model Eğitimi -> Fit yapay zekada eğitim anlamına gelir
model.fit(X_train, Y_train)

# Tahmin
Y_tahmin = model.predict(X_test)

# Tahmin ve Test karşlılaştırması

from sklearn.metrics import confusion_matrix

# Matristeki köşegendeki sayıların toplamı test veriseti boyutuna eşit olmalı,
# Başka alanlarda değerler varsa sapma vardır.
# Burada versicolor'da 2 tanesi virginica'ya kaymış, 9 tanesi doğru
#[[10  0  0]
# [ 0  9  2]
# [ 0  0  9]]

# 2 tane Virginica, araya kaynamış

hata_matrisi = confusion_matrix(Y_test, Y_tahmin)
print(hata_matrisi)

# Hata matrisinin çizimi
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
index = ["setosa","versicolor","virginica"]
columns = ["setosa","versicolor","virginica"]
hata_goster = pd.DataFrame(hata_matrisi,columns,index)
plt.figure(figsize=(10,6))
sns.heatmap(hata_goster, annot=True)