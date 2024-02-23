* model ne kadar karmaşık olursa hata  okadar azalır 
* smote =>veri seti dengesizliği önlemek için yapay veri ekleme

* Gözetimli öğrenme: Regresyon -- sınıflandırma
   * Regresyon: Regresyon, bir bağımlı değişkenin (hedef) bir veya daha fazla bağımsız değişkenle ilişkisini modelleyen ve bu ilişkiyi kullanarak sürekli sayısal tahminlerde bulunan bir gözetimli öğrenme türüdür. Örneğin, ev fiyatlarını tahmin etmek veya hava durumunu öngörmek için kullanılabilir.
   * Sınıflandırma: Sınıflandırma, bir bağımlı değişkenin (hedef) belirli bir kategoriye veya sınıfa ait olup olmadığını belirlemek için kullanılan bir gözetimli öğrenme türüdür. Bu, iki veya daha fazla sınıf arasında ayrım yapmak için kullanılabilir. Örneğin, bir e-postanın spam veya spam olmadığını belirlemek için sınıflandırma kullanılabilir.
* Gözetimsiz öğrenme: Kümeleme
   * Kümeleme: Gözetimsiz öğrenme türlerinden biri olan kümeleme, veri setindeki benzer örnekleri bir araya getirip gruplandırmak için kullanılır. Bu, veri setinde gizli yapıları keşfetmek amacıyla yapılır. Örneğin, müşterileri alışveriş alışkanlıklarına göre gruplandırmak için kümeleme kullanılabilir.


* Cross Validation : Cross Validation, bir makine öğrenimi modelinin performansını değerlendirmek için kullanılan bir tekniktir. Genellikle veri setini eğitim ve test setlerine bölmenin ötesinde, veri setini farklı alt kümelerine bölerek modelin genel performansını değerlendirmek için kullanılır.

   * Overfitting ve Underfitting:
        Overfitting: Model, eğitim verilerine aşırı derecede uyum sağlar, bu da yeni verilere genellenme yeteneğini düşürür.
        Underfitting: Model, eğitim verilerini yeterince öğrenemez ve bu da genelleme yeteneğini düşürür.

   * Hyperparameter ve Parameter:
        Hyperparameter: Modelin yapılandırılmasını etkileyen ve genellikle önceden belirlenen ayarlar (örneğin, öğrenme oranı, epoch sayısı).
        Parameter: Modelin eğitildiği sırasında öğrenilen ağırlıklar ve bias değerleri.

   * Gradient Descent:
        Bir optimizasyon algoritmasıdır ve modelin kaybını (hata) minimize etmek için kullanılır.

   * Feature Engineering:
        Veri setindeki özellikleri (feature) daha anlamlı veya etkili hale getirme süreci.

   * Bias-Variance Tradeoff:
        Modelin karmaşıklığı ve hatası arasındaki dengeyi ifade eder. Daha karmaşık modeller genellikle daha düşük eğitim hatası elde eder, ancak test verilerinde daha yüksek hata riski taşır.

   * Ensemble Learning:
        Birden çok modelin bir araya getirilerek daha güçlü ve stabil bir model oluşturulması.

   * Cross-Validation:
        Model performansını değerlendirmek için veri setinin farklı alt kümelerine bölünerek test edilmesi.

   * Confusion Matrix:
        Sınıflandırma modelinin performansını değerlendirmek için kullanılan bir matris.

   * Precision, Recall, F1-Score:
        Sınıflandırma modellerinin performansını ölçen metrikler.

   * Clustering:
        Veri setindeki benzer örnekleri bir araya getirip gruplandırma işlemi.

   * Dimensionality Reduction:
        Veri setindeki özellik sayısını azaltma süreci, örneğin PCA (Principal Component Analysis).

   * Neural Networks ve Deep Learning:
        Yapay sinir ağları ve derin öğrenme yöntemleri.

   * Reinforcement Learning:
        Bir ajanın çevresiyle etkileşime geçerek öğrenme sürecini ifade eder.

   * Transfer Learning:
        Bir görevde öğrenilen bilgilerin, başka bir görevde kullanılması.

   * Natural Language Processing (NLP):
        Bilgisayarların doğal dil anlama ve işleme yeteneklerini geliştiren bir alan.


CNN RNN DNN LSTM AE Algoritmaları :

* CNN (Convolutional Neural Network - Evrişimli Sinir Ağı):
    Görüntü işleme ve tanıma alanında kullanılan bir sinir ağı türüdür. Evrişimli sinir ağları, özellikle resim verilerinde lokal desenleri tanıma konusunda başarılıdır.

* RNN (Recurrent Neural Network - Tekrarlayan Sinir Ağı):
    Sıralı veriler üzerinde çalışan bir sinir ağı türüdür. RNN'ler, önceki zaman adımlarından elde edilen bilgileri hafızalarında tutarak sıralı veri analizi için uygundur. Ancak, uzun vadeli bağımlılıkları yakalama konusunda zorluklar yaşayabilirler.

* DNN (Deep Neural Network - Derin Sinir Ağı):
    Birbirine bağlı çok katmanlı (derin) bir sinir ağıdır. DNN'ler, karmaşık görevleri öğrenme kapasiteleri ile bilinir. Derin öğrenme alanında yaygın olarak kullanılırlar.

* LSTM (Long Short-Term Memory - Uzun Kısa Vadeli Bellek):
    RNN'lerin geliştirilmiş bir versiyonudur ve özellikle uzun vadeli bağımlılıkları daha etkili bir şekilde ele alabilirler. LSTM hücreleri, önceki zaman adımlarından gelen bilgileri uzun vadeli belleklerinde saklarlar.

* AE (Autoencoder - Otomatik Kodlayıcı):
    Veri setindeki temel yapıyı öğrenen bir tür unsupervised (gözetimsiz) öğrenme algoritmasıdır. Encoder ve decoder olmak üzere iki temel bileşeni vardır. Veriyi düşük boyutlu bir temsil (latent space) ile kodlar ve ardından bu temsili kullanarak orijinal veriyi çıkışa çözer.

Makine öğrenmimi Süreci :

    1) Öğrenme Tercihi:
        Denetimli Öğrenme: Eğitim verileri ile birlikte etiketlenmiş çıkış verileri kullanılarak model eğitilir. Amacı, giriş verileri ile etiket arasındaki ilişkiyi öğrenmektir.
        Denetimsiz Öğrenme: Etiketlenmiş çıkış verileri olmadan sadece giriş verileri kullanılarak model eğitilir. Amacı, veri içerisindeki yapıyı, desenleri ve ilişkileri keşfetmektir.

    2) Model Ayrımı:
        Sınıflandırma: Bir girişin belirli bir sınıfa ait olup olmadığını belirlemek için kullanılır.
        Regresyon: Sayısal değerler tahmin etmek için kullanılır.
        Kümeleme: Veriyi belirli kümeler veya gruplar halinde ayırmak için kullanılır.
        Boyut Azaltma: Giriş verilerini daha düşük boyutlu bir temsile dönüştürmek için kullanılır.

    3) Model Tercihi:
        Seçilen model türüne göre uygun bir algoritma seçilir (örneğin, sınıflandırma için SVM, regresyon için lineer regresyon, kümeleme için K-Means, vb.).
        Modelin öğrenme hiperparametreleri belirlenir ve model eğitilir.

    4) Model Test:
        Eğitilen model, test verileri üzerinde değerlendirilir.
        Modelin performansı ölçülür ve hata analizi yapılır.
        Modelin genelleme yeteneği değerlendirilir.

## Apache Spark ve Hadoop