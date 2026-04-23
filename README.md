# YZM304 Lab Projects

Bu repo, Ankara Universitesi YZM304 Derin Ogrenme dersi kapsaminda hazirlanan proje odevlerini icerir.

## Projects

| Klasor | Konu | Kisa aciklama |
| --- | --- | --- |
| `project-1` | MLP ile tablosal siniflandirma | Heart Failure Clinical Records veri seti uzerinde NumPy, scikit-learn ve PyTorch backend karsilastirmasi. |
| `project-2` | CNN ile goruntu siniflandirma | Digits goruntu veri seti uzerinde iki ozel CNN, AlexNet esinli CNN ve CNN ozellikleri + SVM hibrit model karsilastirmasi. |

Her proje kendi `README.md`, `requirements.txt`, testleri, kaynak kodlari ve uretilmis ciktilari ile ayri olarak tekrar calistirilabilir.

## Calistirma

Proje 1:

```bash
cd project-1
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 -m src.run_all
```

Proje 2:

```bash
cd project-2
python3 -m pip install -r requirements.txt
python3 -m pytest -q
MPLCONFIGDIR=.mplconfig python3 -m src.run_all
```

## Notlar

- Ayrintili raporlar ilgili proje klasorlerinin `README.md` dosyalarindadir.
- Odev PDF dosyalari ilgili proje klasorlerinde `docs/assignment/` altinda tutulur.
- Deney sonuclari `outputs/` altinda tablo, rapor ve gorsel olarak saklanir.
