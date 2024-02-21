# Yapay sinir ağları (YSAs)(SNN-structural neural networks), insan beyninin çalışma prensiplerinden esinlenerek oluşturulan bir makine öğrenimi türüdür. Yapay sinir ağları, öğrenme sürecinde veriye dayalı olarak modelin ağırlıklarını güncelleyerek belirli görevleri öğrenir.

#    Convolutional Neural Networks (CNN):(Evrişimsel Sinir Ağları):
        CNN'ler, genellikle görsel veri analizi ve tanıma görevleri için kullanılır.
        Girdi verisinde lokal desenleri tanımak ve özellikleri çıkarmak için özel bir mimariye sahiptir.
        Görüntü sınıflandırma, nesne tespiti ve yüz tanıma gibi görevlerde başarıyla kullanılırlar.
        Convolutional katmanlar, pooling katmanları ve tam bağlantılı katmanlardan oluşan bir yapıya sahiptir.

#    Recurrent Neural Networks (RNN):(Tekrarlayan Sinir Ağları):
        RNN'ler, zamanla değişen veri ve sıralı bilgilerle iyi başa çıkan bir yapıya sahiptir.
        Dil modelleme, metin oluşturma, çeviri gibi sıralı veri işleme görevlerinde etkilidirler.
        Bellek hücreleri sayesinde geçmiş bilgileri hatırlayabilir, bu da sıralı veri üzerinde daha etkili öğrenme sağlar.
        Ancak, RNN'lerde karşılaşılan zorluklardan biri olan uzun vadeli bağımlılık sorununu çözmek için geliştirilmiş türleri bulunmaktadır, örneğin Long Short-Term Memory (LSTM) ve Gated Recurrent Unit (GRU).
        bir önceki katmanın ağırlıklarını bir sonraki katmana iletemiyor 
        geri beslemede kaybolan gradyan(ağırlık) problemi çıkıyor
    
#    Long Short-Term Memory (LSTM):(Uzun Kısa Süreli Bellek):
        LSTM, özellikle uzun vadeli bağımlılık sorununu çözmek amacıyla tasarlanmış bir RNN hücresidir.
        Standart RNN'lerde, zamanla azalan veya kaybolan gradyanlar nedeniyle uzun vadeli bağımlılıkları takip etme konusunda zorlanırlar.
        LSTM, bu sorunu çözmek için özel bir bellek hücresi yapısına sahiptir. Bu hücre, üç ana kapı (giriş, çıkış ve unutma kapıları) kullanarak bilgi akışını düzenler.
        Giriş kapısı, hangi bilginin güncelleneceğini belirler. Çıkış kapısı, hangi bilginin kullanılacağını kontrol eder. Unutma kapısı ise hangi bilgilerin unutulacağını belirler.
        LSTM, uzun vadeli bağımlılıkları daha etkili bir şekilde öğrenebilir ve saklayabilir.
        metin sınıflandırma metin üretme gibi alanlarda kullanılır

#    Gated Recurrent Unit (GRU):(Kapılı Tekrarlama Ünitesi):
        GRU, LSTM'e benzer bir amaçla geliştirilmiş bir RNN hücresidir ve daha basit bir yapıya sahiptir.
        LSTM'e kıyasla daha az parametreye ve daha az hesaplama karmaşıklığına sahiptir.
        GRU da bir giriş, çıkış ve unutma kapısına sahiptir, ancak LSTM'den farklı olarak, tek bir güncelleme kapısı vardır.
        Bu yapısı sayesinde GRU, benzer performansı sağlarken daha az hesaplama maliyetine sahiptir.
        Hem kısa vadeli hem de uzun vadeli bağımlılıkları öğrenme yeteneğine sahiptir.
        metin sınıflandırma metin üretme gibi alanlarda kullanılır

#     Autoencoder(AE):(Otomatik kodlayıcı):
        Unsupervised learning (denetimsiz öğrenme) için kullanılır.
        Veri setinin temsili (encoding) oluşturarak veriyi sıkıştırır ve bu temsil üzerinden veriyi yeniden oluşturur.
        Özellik çıkarma ve boyutsal azaltma için kullanılır.
        spam testi,duygu analizi , 

#     DBN (Deep Belief Network), 
        derin öğrenme alanında kullanılan bir yapay sinir ağı modelidir. DBN, genellikle unsupervised (denetimsiz) öğrenme görevlerinde, özellik çıkarma ve temsil öğrenme amacıyla kullanılır. Bu model, birkaç katmandan oluşan bir yapay sinir ağı yapısıdır ve öğrenme sürecinde genellikle iki aşamalı bir yaklaşım kullanır.DBN'ler, özellikle sınıflandırma, regresyon, desen tanıma, özellik mühendisliği ve doğal dil işleme gibi birçok uygulama alanında kullanılmıştır.

#    Generative Adversarial Networks (GAN):(Üretken Rekabet Ağları):
        İki ağın (bir üreteç ve bir ayırt edici) rekabet ettiği bir yapay sinir ağı türüdür.
        Yaratıcı içerik üretimi, resim sentezi ve veri artırma gibi alanlarda kullanılır.

#    Feedforward Neural Networks (FNN):(İleri Beslemeli Sinir Ağları):
        En basit yapay sinir ağı türüdür.
        Giriş katmanından çıkış katmanına doğru tek yönlü bir akışa sahiptir.
        Sınıflandırma ve regresyon problemleri için kullanılabilir.

#    Radial Basis Function Networks (RBFN):(Radyal Temel Fonksiyon Ağları):
        Giriş verisini bir dizi radyal taban fonksiyonu ile işleyen bir türdür.
        Genellikle sınıflandırma ve regresyon görevlerinde kullanılır.

#    Self-Organizing Maps (SOM):(Kendi Kendini Düzenleyen Haritalar):
        Topolojik haritalar oluşturarak veriyi kümelemek için kullanılır.
        Özellikle desen tanıma ve veri madenciliği uygulamalarında kullanılır.

#    Hopfield Networks:(Hopfield Ağları:)
        Bir tür reküran yapay sinir ağıdır ve özellikle hafıza ve optimizasyon problemlerinde kullanılır.
        Maksimum enerji durumunu bulma yeteneğine dayalıdır.

# Denetimli Öğrenme (Supervised Learning):
        Bu türde, algoritma eğitim sürecinde etiketli veri setlerini kullanır.
        Etiketler, giriş verileriyle birlikte hedef çıktıları içerir. Yani, her giriş örneğiyle birlikte doğru çıktı bilgisi sağlanır.
        Öğrenme algoritması, veri setindeki örnekler arasındaki ilişkiyi öğrenir ve ardından yeni, görülmemiş giriş verileri için tahminler yapabilir.
        Örnek uygulamalar: sınıflandırma ve regresyon problemleri.

# Denetimsiz Öğrenme (Unsupervised Learning):
        Bu türde, algoritma eğitim sürecinde etiketli veri setleri kullanılmaz. Veri seti sadece giriş verilerini içerir.
        Algoritma, veri setindeki desenleri veya yapıları keşfetmeye çalışır.
        Öğrenme süreci genellikle veri setindeki gizli yapıları ortaya çıkarmaya odaklanır.
        Örnek uygulamalar: kümeleme, boyut azaltma ve yoğunluk tahmini.

# Yarı Denetimli Öğrenme (Semi-Supervised Learning):
        Bu tür, hem etiketli hem de etiketsiz veri setlerini kullanır.
        Genellikle etiketli veri seti küçükken ve etiketsiz veri seti daha büyükken kullanılır.
        Etiketli veri setinden öğrenilen bilgiler, etiketsiz veri setinden tahmin yapmak için kullanılabilir.
        Örnek uygulamalar: dil modellemesi, görüntü sınıflandırması.

# Takviyeli Öğrenme (Reinforcement Learning):
        Bu tür, bir ajanın bir çevre ile etkileşimde bulunarak öğrendiği bir öğrenme paradigmasıdır.
        Ajan, bir görevde belirli bir davranışı optimize etmeye çalışır ve çevreden gelen geri bildirimlere dayanarak öğrenir.
        Öğrenme süreci genellikle ödüller ve cezalar aracılığıyla gerçekleşir.
        Örnek uygulamalar: oyun stratejileri, robot kontrolü.



# Notlar : 
> weka programı tensorboard gibi model ile ilgili ön görü elde etmek için
> deepl tercüme tarafında iyidir
> önceki özelliklerin kullanılması sırasında lstm ve glu çok iyi sonuç verir
> smoth ve gan yöntemi ile veri setine eleamn ekleme yapılabilir