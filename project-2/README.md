# YZM304 Derin Ogrenme Proje Modulu II

## Introduction
Bu proje, YZM304 Derin Ogrenme dersi II. proje odevi icin evrisimli sinir aglari ile goruntu siniflandirma calismasidir. Calismada benchmark nitelikteki `sklearn.datasets.load_digits` veri seti kullanildi. Veri seti 8x8 gri seviye el yazisi rakam goruntulerinden ve 10 siniftan olusur.

Amac, ayni train-validation-test bolmesi uzerinde iki ozel CNN modelini, literaturde yaygin CNN ailelerinden esinlenen daha derin bir modeli ve CNN ozellik cikarimi + klasik makine ogrenmesi hibrit modelini karsilastirmaktir.

Odev metni `docs/assignment/yzm304_proje2_2526.pdf` altinda saklandi.

## Methods
### Veri seti ve on isleme
- Veri seti: scikit-learn digits, 8x8 grayscale handwritten digit images.
- Toplam ornek: `1797`.
- Sinif sayisi: `10`.
- Goruntu boyutu: `1x8x8`.
- Piksel on isleme: piksel degerleri `0-16` araligindan `0-1` araligina normalize edildi.
- Split: `1257` train, `270` validation, `270` test.
- Split stratejisi: stratified random split, seed `42`.

### Modeller
- `LeNetLikeCNN`: LeNet-5 benzeri temel CNN. Conv2d, ReLU, AvgPool2d, Flatten ve fully connected katmanlari acik olarak yazildi.
- `ImprovedLeNetCNN`: Ilk modelle ayni evrisim hiperparametrelerini ve classifier genisliklerini korur; ek olarak batch normalization ve dropout kullanir.
- `AlexNetSmallCNN`: AlexNet/VGG tarzinda daha derin ve max-pooling kullanan kompakt CNN mimarisidir. `torchvision` bagimliligi olmadan acik sinif olarak yazildi.
- `Hybrid_AlexNetFeatures_SVM`: Egitilmis `AlexNetSmallCNN` modelinin feature extractor kismi ile `.npy` ozellik ve label setleri olusturuldu; ardindan RBF kernel SVM egitildi.

### Egitim ayarlari
- Loss function: `torch.nn.CrossEntropyLoss`.
- Optimizer: Adam.
- Learning rate: `0.001`.
- Epoch: `30`.
- Batch size: `64`.

Adam, kucuk veri setinde hizli ve stabil yakinlama verdigi icin tercih edildi. Cross entropy, cok sinifli siniflandirma problemi icin standart kayip fonksiyonudur. Batch size `64`, veri seti kucuk oldugu icin hem hizli egitim hem de yeterli mini-batch cesitliligi saglar.

## Results
### Test metrikleri
| model                      |   accuracy |   balanced_accuracy |   macro_f1 |   test_loss |   best_val_accuracy |
|:---------------------------|-----------:|--------------------:|-----------:|------------:|--------------------:|
| LeNetLikeCNN               |     0.8889 |              0.8875 |     0.8848 |      0.3127 |              0.8815 |
| ImprovedLeNetCNN           |     0.9741 |              0.9735 |     0.9735 |      0.0532 |              0.9741 |
| AlexNetSmallCNN            |     0.9778 |              0.9772 |     0.9774 |      0.0576 |              0.9889 |
| Hybrid_AlexNetFeatures_SVM |     0.9889 |              0.9886 |     0.9887 |    nan      |            nan      |

### Hibrit ozellik setleri
- Train feature shape: `[1527, 512]`.
- Test feature shape: `[270, 512]`.
- Train label shape: `[1527]`.
- Test label shape: `[270]`.
- Kaydedilen dosyalar: `outputs/features/train_features.npy`, `outputs/features/train_labels.npy`, `outputs/features/test_features.npy`, `outputs/features/test_labels.npy`.

### Uretilen ciktilar
- `outputs/tables/model_metrics.csv`
- `outputs/reports/experiment_summary.json`
- `outputs/figures/confusion_matrix_*.png`
- `outputs/figures/learning_curve_*.png`

### Calistirma
```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
MPLCONFIGDIR=.mplconfig python3 -m src.run_all
```

## Discussion
En iyi test balanced accuracy sonucu `Hybrid_AlexNetFeatures_SVM` modeli ile elde edildi. Temel LeNet benzeri model, odevin referans CNN sinifi olarak islev gordu. Batch normalization ve dropout eklenen ikinci model, ayni ana hiperparametreleri koruyarak regularizasyonun etkisini olcmek icin kullanildi. Daha derin AlexNet esinli model ise daha fazla kanal ve max-pooling ile daha zengin uzamsal ozellikler ogrenmeyi hedefledi.

Hibrit modelde CNN'in ozellik cikarim mekanizmasi sabit bir temsil uretmek icin kullanildi ve bu temsil SVM ile siniflandirildi. Bu yaklasim, tam CNN siniflandiricisi ile klasik ML siniflandiricisinin ayni goruntu temsili uzerinde karsilastirilmasini sagladi. Hibrit modelin basarimi, CNN feature extractor tarafindan uretilen temsilin klasik bir karar siniri icin de anlamli bilgi tasidigini gosterir.

Bu deneyler kucuk ve temiz bir benchmark veri setinde yapildigi icin sonuclar hizli tekrar uretilebilir. Daha buyuk bir veri setinde CIFAR-10 veya Kaggle tabanli RGB veri ile veri artirma, pretrained modeller ve daha uzun egitim ek olarak incelenebilir.

## References
- Ankara Universitesi YZM304 Derin Ogrenme dersi II. proje odevi, 2025-2026 Bahar Donemi.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. Gradient-based learning applied to document recognition.
- Krizhevsky, A., Sutskever, I., & Hinton, G. ImageNet classification with deep convolutional neural networks.
- scikit-learn digits dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
