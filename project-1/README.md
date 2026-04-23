# YZM304 Derin Ogrenme Proje Modulu I

## Introduction
Bu proje, Ankara Universitesi YZM304 Derin Ogrenme dersi proje odevini Heart Failure Clinical Records veri seti uzerinde tekrar uretilebilir bir ikili siniflandirma calismasi olarak gerceklestirir. Calisma, laboratuvar gunundeki tek gizli katmanli temel MLP varsayimini yeniden kurar; veri on isleme, mimari genisletme, derinlestirme, L2 regularizasyonu ve veri miktari etkisi uzerinden modeli iyilestirmeyi hedefler.

Veri seti olarak UCI kaynagli `heart_failure_clinical_records_dataset.csv` kullanildi. Veri 299 gozlem ve 12 ozellikten olusur; hedef degisken `DEATH_EVENT` ikili sinif etiketidir. Sinif dagilimi `203 survive (0)` ve `96 death (1)` seklindedir. Bu dengesiz dagilim nedeniyle model seciminde yalnizca accuracy yerine validation balanced accuracy ana kriter olarak kullanildi.

Bu repo ayrica ayni mimari ve ayni baslangic agirliklari ile NumPy, scikit-learn ve PyTorch backend'lerini karsilastirir. Amac, odeve uygun sekilde sadece sonuc vermek degil, ayni zamanda deneyleri savunulabilir ve yeniden calistirilabilir hale getirmektir.

Odev metninin repo icinde izlenebilir kalmasi icin orijinal dosya `docs/assignment/YZM304_Proje_Odevi1_2526.pdf` altinda da saklanmistir.

## Methods
### Veri kaynagi ve bolme stratejisi
- Veri kaynagi: [UCI Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart%2Bfail)
- Ham CSV: [heart_failure_clinical_records_dataset.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv)
- Toplam ornek: `299`
- Ozellik sayisi: `12`
- Siniflar: `survived (0)`, `death (1)`
- Stratified split: `%60 train / %20 validation / %20 test`
- Gercek bolme sayilari: `179 train`, `60 validation`, `60 test`
- Tohum: `42`

### On isleme
- Surekli degiskenler: `age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium`, `time`
- Ikili degiskenler: `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`
- `baseline_raw` kosusunda on isleme uygulanmadi.
- Diger uygun kosularda sadece surekli degiskenler train set uzerinde `StandardScaler` ile fit edilip validation ve test setlerine transform edildi.

### Modelleme varsayimlari
- Temel laboratuvar modeli varsayimi: `12-8-1`, tek gizli katman, `sigmoid`
- NumPy MLP sinifi: ozel ileri yayilim + geri yayilim implementasyonu
- Kayip fonksiyonu: `binary cross entropy`
- Optimizasyon: `full-batch SGD`
- Karar esigi: `0.5`
- Cikis aktivasyonu: `sigmoid`
- Gizli katman aktivasyonlari: `sigmoid` veya `relu`
- L2 regularizasyon: opsiyonel, `deep_scaled_l2` ve veri fraksiyonu deneylerinde kullanildi

### Laboratuvar baseline rekonstruksiyonu
- 13.03.2026 laboratuvar calismasinin devam modeli dogrudan verilmedigi icin temel laboratuvar hatti bu projede `baseline_raw` ve `baseline_scaled` kosullari ile yeniden kuruldu.
- `baseline_raw`, iki katmanli MLP'nin ham veri uzerindeki davranisini gostermek icin tutuldu.
- `baseline_scaled`, ayni temel mimarinin sadece veri on isleme etkisini ayirmak icin kullanildi.

### Baslangic agirliklari ve backend esitleme
- Baslangic agirliklari `data/weights/` altinda mimari bazinda kaydedildi.
- Sigmoid gizli katmanlar ve output katmani icin Xavier-benzeri olcekleme kullanildi.
- ReLU gizli katmanlar icin He-benzeri olcekleme kullanildi.
- Tum bias degerleri `0` ile baslatildi.
- NumPy, sklearn ve PyTorch tekrarlarinda ayni split manifesti, ayni baslangic agirliklari, ayni epoch sayisi ve ayni SGD ailesi kullanildi.
- `%50 / %75 / %100` train alt-kumeleri `data/splits/split_manifest.json` icinde nested olacak sekilde kaydedildi.

### NumPy deneyleri
| Deney | Mimari | Aktivasyon | Scaler | LR | Epoch | L2 | Train fraksiyonu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline_raw` | `12-8-1` | sigmoid | Hayir | `0.05` | `1200` | `0` | `1.0` |
| `baseline_scaled` | `12-8-1` | sigmoid | Evet | `0.05` | `1200` | `0` | `1.0` |
| `wide_scaled` | `12-16-1` | sigmoid | Evet | `0.03` | `1400` | `0` | `1.0` |
| `deep_scaled` | `12-24-12-1` | relu | Evet | `0.01` | `1400` | `0` | `1.0` |
| `deep_scaled_l2` | `12-24-12-1` | relu | Evet | `0.01` | `1400` | `1e-3` | `1.0` |
| `deep_scaled_l2_data50` | `12-24-12-1` | relu | Evet | `0.01` | `1400` | `1e-3` | `0.50` |
| `deep_scaled_l2_data75` | `12-24-12-1` | relu | Evet | `0.01` | `1400` | `1e-3` | `0.75` |

### Model secim kurali
Model secimi validation tarafinda su sirayla yapildi:
1. Validation balanced accuracy en yuksek
2. `n_steps` en dusuk
3. Parametre sayisi en dusuk
4. Validation ROC-AUC en yuksek

Bu secim mantigi dengesiz sinif dagiliminda yalnizca ham accuracy ile karar vermemek icin tercih edildi.

### Backend karsilastirmasi
Asagidaki uc ana kosul NumPy referansi ile birlikte `scikit-learn MLPClassifier` ve `PyTorch nn.Module` kullanilarak tekrar yazildi:
- `baseline_scaled`
- `wide_scaled`
- `deep_scaled_l2`

### Calistirma
```bash
python3 -m pip install --user -r requirements.txt
python3 -m pytest -q
python3 -m src.run_all
```

Uretilen temel ciktilar:
- `outputs/tables/numpy_experiment_metrics.csv`
- `outputs/tables/backend_comparison_metrics.csv`
- `outputs/tables/model_selection.csv`
- `outputs/figures/class_distribution.png`
- `outputs/figures/numpy_learning_curves.png`
- `outputs/figures/data_fraction_comparison.png`
- `outputs/figures/confusion_matrix_selected_numpy.png`
- `outputs/figures/confusion_matrix_selected_sklearn.png`
- `outputs/figures/confusion_matrix_selected_pytorch.png`
- `outputs/figures/confusion_matrix_baseline_scaled_numpy.png`
- `outputs/figures/confusion_matrix_baseline_scaled_sklearn.png`
- `outputs/figures/confusion_matrix_baseline_scaled_pytorch.png`
- `outputs/figures/confusion_matrix_wide_scaled_numpy.png`
- `outputs/figures/confusion_matrix_wide_scaled_sklearn.png`
- `outputs/figures/confusion_matrix_wide_scaled_pytorch.png`
- `outputs/figures/confusion_matrix_deep_scaled_l2_numpy.png`
- `outputs/figures/confusion_matrix_deep_scaled_l2_sklearn.png`
- `outputs/figures/confusion_matrix_deep_scaled_l2_pytorch.png`
- `outputs/reports/experiment_summary.md`
- `outputs/reports/experiment_summary.json`
- `outputs/reports/selected_model_report.md`
- `outputs/reports/traceability_matrix.md`

## Results
### NumPy deney ozeti
| Deney | Val balanced acc | Val acc | Test acc | Test F1 | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| `baseline_raw` | `0.5000` | `0.6833` | `0.6833` | `0.0000` | `0.5000` |
| `baseline_scaled` | `0.8216` | `0.8333` | `0.8167` | `0.6667` | `0.8703` |
| `wide_scaled` | `0.8601` | `0.8667` | `0.8167` | `0.6667` | `0.8549` |
| `deep_scaled` | `0.8074` | `0.8333` | `0.8333` | `0.6667` | `0.8511` |
| `deep_scaled_l2` | `0.8074` | `0.8333` | `0.8333` | `0.6667` | `0.8511` |
| `deep_scaled_l2_data50` | `0.7933` | `0.8333` | `0.8333` | `0.7059` | `0.8614` |
| `deep_scaled_l2_data75` | `0.7933` | `0.8333` | `0.8000` | `0.6000` | `0.8498` |

### Secilen model
- Secilen model: `wide_scaled`
- Mimari: `12-16-1`
- Aktivasyon: `sigmoid`
- On isleme: sadece surekli kolonlarda standardizasyon
- Validation accuracy: `0.8667`
- Validation balanced accuracy: `0.8601`
- Test accuracy: `0.8167`
- Test balanced accuracy: `0.7529`
- Test F1: `0.6667`
- Test ROC-AUC: `0.8549`

`deep_scaled` ve `deep_scaled_l2` kosullari test accuracy tarafinda `0.8333` ile biraz daha yuksek gorunse de secim validation balanced accuracy odakli yapildigi icin `wide_scaled` one cikti. Bu, odevde istenen model secim mantigini dengesiz veri uzerinde savunulabilir hale getirdi.

### Backend karsilastirmasi
| Deney | Backend | Test acc | Test balanced acc | Test ROC-AUC |
| --- | --- | --- | --- | --- |
| `baseline_scaled` | NumPy | `0.8167` | `0.7529` | `0.8703` |
| `baseline_scaled` | sklearn | `0.8167` | `0.7529` | `0.8703` |
| `baseline_scaled` | PyTorch | `0.8167` | `0.7529` | `0.8703` |
| `wide_scaled` | NumPy | `0.8167` | `0.7529` | `0.8549` |
| `wide_scaled` | sklearn | `0.8167` | `0.7529` | `0.8549` |
| `wide_scaled` | PyTorch | `0.8167` | `0.7529` | `0.8549` |
| `deep_scaled_l2` | NumPy | `0.8333` | `0.7510` | `0.8511` |
| `deep_scaled_l2` | sklearn | `0.8333` | `0.7510` | `0.8511` |
| `deep_scaled_l2` | PyTorch | `0.8333` | `0.7510` | `0.8524` |

Backend sonuclari pratik olarak esdegerdir. Bu da ayni split, ayni agirliklar ve ayni optimizer ailesiyle uc farkli implementasyonun birbirini dogruladigini gosterir.

## Discussion
Bu calismada en net kazanc veri on islemeden geldi. Ham veri ile egitilen `baseline_raw` modeli validation balanced accuracy tarafinda `0.5000` de kalirken, ayni laboratuvar tabanli mimarinin standardize edilmis surumu `0.8216` seviyesine cikti. Bu fark, ozellikle `platelets` ve `creatinine_phosphokinase` gibi farkli olceklerde gelen degiskenlerde standardizasyonun kritik oldugunu gosterdi.

Mimari genisletme derinlestirmeye gore daha iyi bir secim verdi. `wide_scaled` validation balanced accuracy tarafinda en iyi model olurken, iki gizli katmanli derin modeller validation dengesini ayni seviyede koruyamadi. Bu veri setinde daha buyuk kapasite otomatik olarak daha iyi secim anlamina gelmedi.

L2 regularizasyonu bu kurulumda validation ve test metriklerini anlamli bicimde yukari tasimadi. `deep_scaled` ve `deep_scaled_l2` neredeyse ayni performansi verdi. Veri miktari deneylerinde `%50` train verisi ile egitilen modelin test F1 skoru gorece iyi kalsa da validation balanced accuracy dusuk oldugu icin secilen model olmadi.

Sonuc olarak, odeve uygun en dengeli cozum `wide_scaled` oldu: tek gizli katman korunurken kapasite artisiyla validation dengesi iyilesti, backend tekrarlarinda ayni sonuc korundu ve tum artefact'lar tekrar uretilebilir sekilde kaydedildi. Gelecek calismalarda mini-batch SGD, batch normalization veya farkli karar esigi secimleri ayrica incelenebilir.
