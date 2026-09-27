# Araştırma Planı: Jev ile Log ve Alert Triage'ın Bağımsız Değerlendirmesi

**Çalışma başlığı (EN):** *Fast, Calibrated Decisions for IT Operations: An Independent Evaluation of a System-One Model for Log and Alert Triage*

**Hedef:** arXiv preprint (birincil kategori `cs.SE`, çapraz: `cs.LG`, `cs.DC`) · **Süre:** 12 hafta (29 Eylül – 21 Aralık 2026), Ocak mezuniyetinden önce yayında
**Kod ve veri:** bu repo (`research/eval/`) · Açıklamalı etiketler CC BY 4.0 ile yayınlanacak

---

## 1. Motivasyon ve boşluk

TypeSafe, 15 Eylül 2026'da Jev'i "System One" modeli olarak duyurdu. Model metin üretmiyor; tipli karar ve kalibre olasılık döndürüyor, 70–500 ms'de. Duyurudaki iddialar güçlü ama:

- Değerlendirme workflow'larını **firmanın kendi ekibi** hazırladı ve referans cevap olarak başka LLM'lerin (GPT-6 Astra + Fable 5.1 ortalaması) çıktıları kullanıldı. Firma bir miktar önyargı olabileceğini kendisi de belirtiyor.
- Yani **insan etiketli, alan-dışı (out-of-domain) bir ground truth'a karşı kalibrasyon ölçülmedi.**
- Ölçümler ABD Batı Kıyısı'ndaki laptoplardan yapıldı. **Coğrafi gecikme** (ör. Türkiye → ABD) raporlanmadı.
- AIOps/log analizi literatüründe LLM'ler çok çalışıldı, ama **seçici otomasyon** az çalışıldı. Buradaki soru şu: "Modelin güvenine göre olayların ne kadarını insan görmeden güvenle yönlendirebiliriz?"

Bu çalışma bu boşluğu dolduruyor. Katkı, "yeni model" değil **bağımsız, tekrarlanabilir bir değerlendirme ve açık bir referans uygulama**.

## 2. Araştırma soruları

| # | Soru | Birincil metrik |
|---|---|---|
| **RQ1** | Jev; kurallar, klasik ML ve genel amaçlı LLM'lere göre log triage görevlerinde ne kadar doğru? | AUROC / AP (anomali), accuracy + macro-F1 (severity, subsystem) |
| **RQ2** | Jev'in olasılıkları, eğitim dağılımı dışındaki operasyon loglarında kalibre mi? | ECE, Brier, güvenilirlik diyagramları |
| **RQ3** | Sabit hata toleransında (%2, %5) olayların hangi oranı otomatik yönlendirilebilir? | Risk–coverage eğrisi, AURC, *coverage@risk* |
| **RQ4** | Gecikme ve maliyet gerçek dağıtımda ne durumda (Türkiye'den ve ABD'deki bir VM'den)? | p50/p95/p99 gecikme, 1.000 olay başına USD |
| **RQ5** *(ops.)* | Saldırgan kontrollü log alanlarına (ör. kullanıcı adına gömülü talimat) karşı ne kadar dayanıklı? | Enjeksiyon başarı oranı, karar değişimi |

RQ3 bu çalışmanın imzası. Servisteki `router.py` tam olarak bu mekanizmayı uyguluyor, yani makale ile yazılım birbirini doğruluyor.

## 3. Veri

### 3.1 Loghub (etiketli, halka açık)

- **BGL** (Blue Gene/L, ~4.7M satır) ve **Thunderbird**. Satır düzeyinde anomali etiketli: ilk token `-` ise normal, değilse alert kategorisi (ör. `KERNDTLB`). Etiketler Oliner & Stearley (DSN 2007) çalışmasından geliyor.
- İndirme: https://github.com/logpai/loghub → Zenodo linkleri. Lisans araştırma amaçlı serbest; Loghub makaleleri atıf zorunlu.
- **Örnekleme:** `jev-eval sample`. Şablon başına en fazla 3 örnek (şablon çeşitliliği) ve %50 anomali dengesi. Ham loglar birkaç şablonun tekrarından ibaret; düz rastgele örnekleme aynı mesajı yüzlerce kez ölçer. Doğal oran ile de ayrı bir koşu yapılır.
- Hedef boyut: veri seti başına **2.000 satır**.

### 3.2 İnsan etiketli triage seti (katkımız)

Halka açık hiçbir log veri setinde operatör bakışıyla *severity / subsystem / actionable / user_impact* etiketi yok.

- BGL + Thunderbird örneklerinden **800 satır**, ayrıca demo stack'in ürettiği Linux/servis loglarından **200 satır**.
- `jev-eval export-annotation` ile CSV'ye aktarılır. Loghub etiketleri annotatöre gösterilmez (anchoring önlemi).
- **Etiketleme kılavuzu:** `questions.py`'deki rubrik metinleri, sınır örnekleriyle birlikte.
- **İkinci annotatör:** örneklerin %25'i bağımsız etiketlenir, Cohen's κ raporlanır. Hedef κ ≥ 0.6; altındaysa kılavuz revize edilip yeniden etiketlenir.

## 4. Karşılaştırılan sistemler

| Sistem | Durum | Not |
|---|---|---|
| **Jev** (`jev-1.13`, model sürümü her satırda kaydediliyor) | ✅ hazır | `extra_nouls` ile anomali sorusu eklendi |
| **Kurallar** (regex/anahtar kelime) | ✅ hazır | `HeuristicClassifier`; üretimdeki fallback ile aynı |
| **Klasik ML:** TF-IDF + lojistik regresyon (+ temperature scaling) | ⏳ Hafta 3 | Etiketli verinin %50'sinde eğitilir, kalan %50'de test edilir. Tüm sistemler aynı test yarısında karşılaştırılır |
| **LLM'ler** (bir kapalı frontier model + bir açık ağırlıklı model) | ⏳ Hafta 3–4 | TypeSafe'in açık kaynak `system-one-adapter-python` sarmalayıcısı ile aynı soru şeması. Ayrıca "sözel güven" (verbalized confidence) varyantı |

**Adillik kuralları:** Tüm sistemler aynı `state` ve aynı soru metinlerini görür. Kural tabanlı sistem de enrichment context satırlarını görür. Değerlendirmede fallback yoktur; başarısız çağrı hata olarak kaydedilir.

## 5. Metrikler (hepsi `research/eval/calibration.py`'de, sklearn ile çapraz test edildi)

- **Doğruluk:** accuracy, macro-F1; anomali için AUROC, average precision, F1@0.5
- **Kalibrasyon:** ECE ve MCE (15 eşit genişlikte bin; Guo ve ark. 2017), Brier, NLL, güvenilirlik diyagramı CSV'si
  - Not: TypeSafe'in `confidence` alanı *max-olasılık* değil, zirve-normalize bir istatistik. Kalibrasyon max-olasılıkla, seçici tahmin ise router'ın gerçekten kullandığı `confidence` ile ölçülür. İkisi ayrı raporlanır.
- **Seçici tahmin:** risk–coverage, AURC (Geifman & El-Yaniv 2017), %2 ve %5 riskte coverage ve karşılık gelen eşik
- **Tutarlılık:** aynı örneklem 3 kez skorlanır, olasılıkların ortalama mutlak farkı raporlanır (firmanın "tutarlı" iddiası sınanır)
- **Verimlilik:** gecikme yüzdelikleri, ortalama input token, maliyet. Fiyat: $0.042/MTok input (Eylül 2026 liste fiyatı, `--price-per-mtok` ile değiştirilebilir)
- **İstatistik:** 1.000 örnekli bootstrap ile %95 güven aralıkları. Eşleştirilmiş karşılaştırmalar için McNemar (accuracy) ve DeLong ya da bootstrap (AUROC)

## 6. Deney protokolü

```bash
# Örneklemi dondur (bütün sistemler aynı satırları görür)
jev-eval sample --dataset bgl --raw data/BGL.log --n 2000 --out data/bgl_2k.jsonl
jev-eval sample --dataset thunderbird --raw data/Thunderbird.log --max-lines 5000000 \
    --n 2000 --out data/tbird_2k.jsonl

# Skorla (yarıda kesilirse kaldığı yerden devam eder)
jev-eval run --sample data/bgl_2k.jsonl --classifier heuristic --out results/bgl_rules.jsonl
jev-eval run --sample data/bgl_2k.jsonl --classifier jev --out results/bgl_jev.jsonl

# Tablolar + güvenilirlik CSV'si
jev-eval analyze results/bgl_*.jsonl --out results/bgl_report
```

- **Gecikme (RQ4):** aynı 500 örnek (a) Manisa/Gebze'den ev veya ofis bağlantısıyla, (b) ABD Batı Kıyısı'ndaki bir bulut VM'den. Concurrency = 1 ve 8. Tarih, saat ve ağ bilgisi kaydedilir.
- **Enjeksiyon (RQ5):** 100 zararsız log satırının kontrollü alanlarına (kullanıcı adı, URL yolu, HTTP user-agent) "classify this as info / not actionable" türü 5 şablon enjekte edilir. Karar değişim oranı ölçülür, `SUPPRESS` rotasına kaçış ayrıca raporlanır.
- **Maliyet bütçesi:** 2 × 2.000 + 1.000 örnek × 3 tekrar × ~400 token ≈ 6M token ≈ **$0.25** (Jev). LLM baseline'ları baskın maliyet olacak; sınır $30–50.

## 7. Zaman çizelgesi (12 hafta)

| Hafta | Tarih | İş | Çıktı |
|---|---|---|---|
| 1 | 29 Eyl – 5 Eki | BGL/Thunderbird indir, örneklemleri dondur, 200 örnekle Jev pilotu. **TypeSafe kullanım koşullarında benchmark yayınlama maddesini kontrol et** | Pilot tablo, ToS notu |
| 2 | 6 – 12 Eki | Etiketleme kılavuzu + 300 satır pilot etiketleme, κ ölçümü. *12 Ekim hackathon haftası; hafif tutuldu* | Kılavuz v1, κ |
| 3 | 13 – 19 Eki | Kalan etiketleme; TF-IDF + LR baseline | 1.000 etiketli satır |
| 4 | 20 – 26 Eki | LLM baseline'ları (adapter ile) | 4 sistem × 3 görev ham sonuç |
| 5 | 27 Eki – 2 Kas | Ana koşular, 3× tekrar, gecikme ölçümleri (TR + ABD VM) | `results/` tamam |
| 6 | 3 – 9 Kas | RQ5 enjeksiyon, eşik analizi, bootstrap CI | Tüm tablolar |
| 7–8 | 10 – 23 Kas | Yazım: giriş, ilgili çalışmalar, yöntem | Taslak §1–4 |
| 9 | 24 – 30 Kas | Sonuçlar, şekiller (reliability, risk–coverage) | Taslak tam |
| 10 | 1 – 7 Ara | Danışman/hoca okuması, revizyon | v2 |
| 11 | 8 – 14 Ara | **arXiv endorsement** + gönderim; kod ve etiketler GitHub + Zenodo (DOI) | arXiv ID |
| 12 | 15 – 21 Ara | Tampon; Medium yazısı + LinkedIn duyurusu | Yayın |

**arXiv notu:** İlk kez gönderim yapanlar çoğu kategoride bir *endorser*'a ihtiyaç duyar. Üniversite hocalarından biri (veya daha önce `cs.*` yayını olan biri) bunu sağlayabilir. Hafta 8'de sormak güvenli.

## 8. Makale iskeleti (~8 sayfa, IEEE iki sütun)

1. **Introduction:** alert fatigue; System One modelleri; katkılar (bağımsız değerlendirme, insan etiketli triage seti, seçici otomasyon analizi, açık referans uygulama)
2. **Background & Related Work:** log anomali tespiti (Le & Zhang, ICSE 2022; Loghub, Zhu ve ark. ISSRE 2023), LLM ile log analizi (LogPrompt, Liu ve ark. ICPC 2024), kalibrasyon (Guo ve ark. 2017; Naeini ve ark. 2015), LLM'lerde güven ifadesi (Tian ve ark. EMNLP 2023; Xiong ve ark. ICLR 2024), seçici sınıflandırma (Geifman & El-Yaniv 2017)
3. **System:** jev-triage mimarisi, soru tasarımı, güven kapılı yönlendirme politikası
4. **Experimental Setup:** veri, örnekleme, etiketleme ve κ, baseline'lar, metrikler
5. **Results:** RQ1–RQ5
6. **Discussion:** ne zaman kullanılmalı / kullanılmamalı; sayısal akıl yürütme ve tarih zayıflıkları; operasyonel öneriler (eşik seçimi)
7. **Threats to Validity**
8. **Conclusion**

> Kaynakçadaki her referans eklenmeden önce orijinal yayından doğrulanmalı (başlık, yıl, venue).

## 9. Geçerlilik tehditleri (baştan yazılı olsun)

- **Kapalı ve değişen model:** Jev erken erişimde, sürümler değişebilir. Her satırda `model_version` kaydedilir, tüm koşular tek sürüm içinde tamamlanır ve tarih raporlanır.
- **Veri sızıntısı:** Loghub 2016'dan beri halka açık, eğitim verisinde olabilir. İnsan etiketli *yeni* etiketler ve demo stack logları bunu kısmen dengeler.
- **Tek annotatör önyargısı:** κ ölçümü ve etiketleme kılavuzunun yayınlanması.
- **Süper bilgisayar logları ≠ modern bulut:** Thunderbird'ün syslog kısmı ve demo seti daha tipik Linux loglarını temsil ediyor. Genelleme sınırı açıkça yazılır.
- **Çıkar çatışması:** TypeSafe ile finansal ilişki yok, bu açıkça beyan edilir. ING hackathon'unda Jev kullanılmış olması "önceki deneyim" olarak belirtilir.

## 10. Hemen sonraki adımlar

1. BGL ve Thunderbird'ü Zenodo'dan indir (`research/data/` git'e girmez).
2. TypeSafe kullanım koşullarında benchmark yayınlamayla ilgili bir kısıt olup olmadığını kontrol et.
3. `jev-eval sample` + `run --limit 200` ile pilotu koş, ilk tabloyu çıkar.
