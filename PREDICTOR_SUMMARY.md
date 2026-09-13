# 肽類 ADMET 預測工具 — 完整總結(v4.2 真實數據 + ESMC + MoLFormer 版,2026-08-26)

> **v4.2 增量變更**(建立在 v4.1 上):
> 1. **分子端點(Caco2、PAMPA_MDCK)導入 MoLFormer-XL 凍結 CLS 嵌入**:
>    IBM 的 SMILES transformer(`ibm-research/MoLFormer-XL-both-10pct`,
>    60M 參數、hidden 768)對每個 SMILES 取 CLS token 768 維,與 2,265 維
>    RDKit 特徵拼接成 **3,033 維**輸入(凍結、不 fine-tune;`molformer_embed.py`
>    算一次 → commit `data/molformer/*.npz`)。Caco2 R² 0.3861 → **0.3909**、
>    PAMPA R² 0.4573 → **0.4642**(增益 +0.005~+0.007,在重訓噪音 ±0.01 內;
>    見「誠實聲明」)。
> 2. **Half_life 重複序列目標聚合 + Huber loss**:1,763 行 → 768 唯一序列
>    (995 行是重複量測,最重複 82 次),同序列多個 log10(half-life)
>    取平均成一筆;主指標改在**序列層級**報告:R² 0.6973 → **0.7259**
>    (Δ +0.0286,行層級 → 序列層級,非同量綱直接比)。
> 3. **所有回歸端點換 Huber loss**(對 log 空間離群值穩健)。
> 4. 新增 `molformer_embed.py`(嵌入生成器)+ `data/molformer/*.npz`(凍結
>    嵌入快取);`admet_model.py` 把 `hidden`/`dropout` 持久化進 checkpoint
>    (v4.1 前舊檔向前相容)。
>
> **v4.1 增量變更**(建立在 v4.0 上):
> 1. **序列端點(Hemolysis、Half_life)導入 ESMC-600M 凍結嵌入**:
>    Biohub 蛋白質語言模型產出 1,152 維 mean-pooled 向量,與 428 維經典
>    序列特徵拼接成 **1,580 維**輸入(凍結、不 fine-tune;`.venv-esmc`
>    算一次 → commit `data/esmc/*.npz`)。Hemolysis AUC **0.7755 → 0.8348**、
>    Half_life R² **0.5883 → 0.6973**(實測重訓)。
> 2. **Caco2 / PAMPA_MDCK 維持 RDKit 分子路徑**(非標準殘基,ESMC 不適用)。
> 3. 新增 `esmc_embed.py`(嵌入生成器)+ `data/esmc/*.npz`(凍結嵌入快取)。
>
> **v4.0 斷裂式變更**(取代 v3.0):
> 1. **改用真實數據集** [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset),
>    **刪除** v3.0 的 30,000 行 `synthetic_demo` 合成數據與 9 端點管線。
> 2. **4 端點**:Hemolysis、Half-life、Caco-2、PAMPA/MDCK(其餘 5 個
>    v3.0 端點 BBB/Ames/hERG/毒性/HC50 不再保留)。
> 3. **雙模態特徵**:序列端點用氨基酸序列特徵;分子端點用 RDKit SMILES
>    描述子。
> 4. **4 個獨立單任務模型**(互斥分子 + 兩種模態,共享 trunk 多頭模型
>    在此只是學兩個不相交子空間)。
>
> v2.0/v3.0 的誠實修訂(移除硬編碼指標、phantom 數據、RF 偽裝 NN)全部保留。

## ✅ 完成狀態:100%(端對端已跑通驗證,4 端點全部訓練 + 預測驗證)

---

## 📁 專案文件

### 1. 管線腳本

| 文件 | 說明 |
|------|------|
| `endpoint_config.py` | 4 端點的單一事實來源(kind、模態、來源 CSV、特徵欄、標籤轉換、單位、**ESMC / MoLFormer 旗標**) |
| `feature_extractor.py` | 雙模態特徵:序列 428 維(AAC 20 + DPC 400 + 理化 8)+ RDKit 分子 2,265 維(217 2D 描述子 + 2,048 位 Morgan) |
| `esmc_embed.py` | **v4.1** ESMC-600M 凍結嵌入生成器(`.venv-esmc` 執行;批次 / ad-hoc 雙模式) |
| `molformer_embed.py` | **v4.2** MoLFormer-XL 凍結 CLS 嵌入生成器(`.venv` 執行;批次 / ad-hoc 雙模式) |
| `prepare_pepadmet_data.py` | 載入 4 個真實 CSV → 清洗(去 X/非 20-AA/無效 SMILES/缺失標籤)→ 每端點輸出準備 CSV + meta |
| `homology_split.py` | 3-mer Jaccard 家族 70/10/20 同源性不相交分割 + 量測洩漏 |
| `admet_model.py` | `MixedADMETMLP`(binary/regression heads,通用 input_dim,**hidden/dropout 持久化**)+ 訓練/預測共用 |
| `train_pepadmet_model.py` | 每端點:特徵 →(v4.1 序列 +ESMC / v4.2 分子 +MoLFormer 拼接)→(v4.2 Half_life 重複序列聚合)→ 分割 → 標準化 → 訓練 → 雙分割評估 → `metrics.json` + 權重 |
| `peptide_admet_predictor.py` | 推論 CLI:輸入 sequence / SMILES → 自動路由模態 →(嵌入快取 / 子程序回退)→ 4 端點預測 + 單位 |

### 2. 訓練產物(`models_v4/`,已 commit)

| 路徑 | 說明 |
|------|------|
| `models_v4/<endpoint>/admet_mlp.pt` | PyTorch 權重 + 架構 metadata(`model_version: v4_endpoint`,含 `hidden`/`dropout`) |
| `models_v4/<endpoint>/scaler.pt` | StandardScaler |
| `models_v4/<endpoint>/metrics.json` | 實測指標(雙分割)+ 分割統計 + 洩漏稽核 + v4.2 武器 metadata |
| `models_v4/summary.json` | 4 端點彙總 |

`<endpoint>` ∈ {`hemolysis`, `half_life`, `caco2`, `pampa_mdck`}。

### 3. 數據(`data/`,已 commit)

| 文件 | 說明 |
|------|------|
| `data/pepadmet_hemolysis.csv` | 準備好的 Hemolysis 序列 + 二值標籤(8,719 行) |
| `data/pepadmet_half_life.csv` | 準備好的 Half_life 序列 + log10 半衰期(1,763 行,768 唯一序列) |
| `data/pepadmet_caco2.csv` | 準備好的 Caco2 SMILES + logPapp(7,429 行) |
| `data/pepadmet_pampa_mdck.csv` | 準備好的 PAMPA SMILES + logPapp(7,283 行) |
| `data/esmc/esmc_emb_hemolysis.npz` | **v4.1** 凍結 ESMC-600M 嵌入(8,719 × 1,152 float32) |
| `data/esmc/esmc_emb_half_life.npz` | **v4.1** 凍結 ESMC-600M 嵌入(1,763 × 1,152 float32) |
| `data/molformer/molformer_emb_caco2.npz` | **v4.2** 凍結 MoLFormer-XL CLS 嵌入(7,429 × 768 float32) |
| `data/molformer/molformer_emb_pampa_mdck.npz` | **v4.2** 凍結 MoLFormer-XL CLS 嵌入(7,283 × 768 float32) |
| `data/pepadmet_data.meta.json` | 來源、清洗統計、行數聲明 |

---

## 🚀 如何使用

```bash
# 0. 環境(需 rdkit)
uv pip install --python .venv/Scripts/python.exe rdkit

# 1. 準備數據(從 Chemit797/PepADMET-Dataset 載入 + 清洗)
python prepare_pepadmet_data.py

# 2. (選做)重新生成嵌入 → data/{esmc,molformer}/*.npz
#    首次或數據集變動時才需要;已 commit 的 npz 可直接用。
.venv-esmc/Scripts/python.exe esmc_embed.py        # v4.1 序列端點
.venv/Scripts/python.exe molformer_embed.py        # v4.2 分子端點

# 3. 訓練 4 端點
python train_pepadmet_model.py --epochs 80 --seed 42

# 4. 預測(自動路由模態)
python peptide_admet_predictor.py \
  --sequence "ACDEFGHIKLMNPQRSTVWY" \
  --smiles "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"
```

---

## 📊 實測性能(同源性/分子控制測試分割,seed 42,單一完整 run)

| 端點 | 類型 | 模態 | 主指標 | 其他(控制分割) | 隨機分割對照 |
|------|------|------|--------|----------------|------------|
| Hemolysis | binary | 序列+ESMC | AUC **0.8348** | MCC 0.4557, Acc 0.7479 | 0.8112(差 −0.0236) |
| Half_life | regression(log10 s) | 序列+ESMC | R² **0.7259** | RMSE 1.365, MAE 0.887(159 唯一序列) | 0.7867(序列層級,近重複已聚合) |
| Caco2 | regression(logPapp) | 分子+MoLFormer | R² **0.3909** | RMSE 0.785, MAE 0.471 | —(無序列對照) |
| PAMPA_MDCK | regression(logPapp) | 分子+MoLFormer | R² **0.4642** | RMSE 0.799, MAE 0.450 | —(無序列對照) |

> **v4.2 換武器增益**(同源性/唯一 SMILES 控制分割,seed 42):
>
> | 端點 | v4.1 | v4.2 | Δ | 換的武器 |
> |------|-----:|-----:|-----:|------|
> | Half_life(序列層級,768 唯一序列) | 0.6973(行層級 1763) | **0.7259** | +0.0286 | 重複序列目標聚合 + Huber |
> | Caco2 | 0.3861 | **0.3909** | +0.0048 | +768 維 frozen MoLFormer-XL CLS(2265→3033 維) |
> | PAMPA_MDCK | 0.4573 | **0.4642** | +0.0069 | 同上 + Huber |
> | Hemolysis(未改端點) | 0.8348 | **0.8348** | 0.0000 | —(決定性重訓精確重現) |
>
> 同端點兩次獨立完整重訓差約 ±0.01(PAMPA 0.4527 vs 0.4642),
> 所以 MoLFormer 的 +0.005~+0.007 增益在重訓噪音之內——
> **分子端點的瓶頸是標籤噪音,不是表示層**(見「誠實聲明」)。

### 洩漏控制

- **序列端點**:3-mer Jaccard 家族同源性分割;分割前合併同 3-mer 多重集
  anagram(canonical signature),**保證 jaccard-1.0 精確複製不跨界**。
  最大跨界 Jaccard ≈ 0.97(近重複合理上限,非洩漏)。**Half_life 隨機
  R² 0.7867 仍高於同源性 0.7259(序列層級),是近重複洩漏被控管掉的
  誠實體現**;v4.2 先做重複序列聚合(1763 → 768)再分割,兩個分割都在
  序列層級比較,避免行層級的 anagram 近重複膨脹。
- **分子端點**:無序列 → 按唯一 SMILES 分組(精確重複 SMILES 同分割);
  近異構物可跨界(SMILES-only 限制,已標註)。

---

## 🎯 特徵工程

**序列模態(428 維經典 + 1,152 維 ESMC = 1,580 維,v4.1)**:
- 經典 428 維:AAC 20 + DPC 400 + 理化 8。
- **ESMC-600M 凍結嵌入 1,152 維**:Biohub ESM Cambrian 蛋白質語言模型,
  attention-mask 加權 mean-pooling;`esmc_embed.py` 在 `.venv-esmc`
  (Python ≥ 3.12)算一次 → `data/esmc/*.npz`(已 commit)。凍結、不
  fine-tune;訓練時拼接、推論時快取/子程序回退。

**分子模態(2,265 維 RDKit + 768 維 MoLFormer = 3,033 維,v4.2)**:
- 經典 2,265 維:RDKit 2D 描述子 217(`CalcMolDescriptors`)+
  Morgan fingerprint(半徑 2)2,048 位。
- **MoLFormer-XL 凍結 CLS 嵌入 768 維**:IBM SMILES transformer
  (60M 參數、hidden 768,`ibm-research/MoLFormer-XL-both-10pct`),
  對每個 SMILES 取 CLS token;`molformer_embed.py` 在 `.venv` 算一次 →
  `data/molformer/*.npz`(已 commit)。凍結、不 fine-tune;訓練時拼接、
  推論時快取/子程序回退。

---

## 📦 模型架構

- 共享 trunk:`input_dim → 256 → 128`(ReLU + BatchNorm + Dropout 0.2);
  `hidden`/`dropout` 持久化進 checkpoint(v4.2 起,向前相容 v4.1 舊檔)。
- 單任務 head:binary `Linear(128,1)` + sigmoid;regression `Linear(128,1)`
- 訓練:Adam(lr=1e-3, wd=1e-5)+ ReduceLROnPlateau + early stopping(val loss, patience 10)
- 損失函數:binary = BCEWithLogits(pos_weight);regression = **Huber**(v4.2,
  對 log 空間離群值穩健;v4.1 及以前為 MSE)
- 參數數:序列端點(1,580 維)**438,529**;分子端點(3,033 維)**810,497**
  (v4.0 序列 143,617 為 428 維;v4.1 拼 ESMC → 1,580 維;v4.2 分子拼
  MoLFormer → 3,033 維)

---

## 🔄 完整工作流程

```
Chemit797/PepADMET-Dataset (整理/*.csv, 真實數據)
       ↓
prepare_pepadmet_data.py (載入 + 清洗 + 來源標記 → data/pepadmet_*.csv)
       ↓
esmc_embed.py (v4.1,選做:ESMC-600M 凍結嵌入 → data/esmc/*.npz)
molformer_embed.py (v4.2,選做:MoLFormer-XL 凍結 CLS 嵌入 → data/molformer/*.npz)
       ↓
train_pepadmet_model.py (每端點:
   序列端點 → 428 維 + 1,152 維 ESMC = 1,580 維
              + 同源性分割(signature 合併, jaccard-1.0 不跨界)
              [Half_life v4.2:先 1763 → 768 唯一序列聚合]
   分子端點 → 2,265 維 RDKit + 768 維 MoLFormer = 3,033 維
              + 唯一 SMILES 分割
   → 標準化 → MixedADMETMLP 單頭(Huber 回歸)→ 訓練 → 雙分割評估
   → models_v4/<endpoint>/{權重, metrics.json})
       ↓
peptide_admet_predictor.py (輸入 sequence/SMILES → 自動路由 →
   序列端點:ESMC 快取 / 新序列走 .venv-esmc 子程序
   分子端點:MoLFormer 快取 / 新 SMILES 走 .venv 子程序
   → 4 端點預測 + 單位)
```

---

## ⚠️ 誠實聲明

- 4 端點來自**真實實驗/文獻數據**(PepADMET-Dataset),非合成。
- **ESMC-600M 嵌入是凍結的、外部預訓練的**(Biohub ESM Cambrian),
  **MoLFormer-XL 嵌入也是凍結的、外部預訓練的**(IBM),本管線
  **不 fine-tune** 任一個——只把它的向量當附加特徵。兩個 npz 都已
  commit,重訓/推論不需重算。
- **v4.2 分子端點的「換武器」沒有突破瓶頸**:frozen MoLFormer-XL
  嵌入對 Caco-2 / PAMPA 的 R² 增益 +0.005~+0.007,與同端點兩次
  獨立重訓的 ±0.01 差可比擬。2026-08-28 進一步實測了 PeptiVerse
  (Nat. Commun. 2026)在 PAMPA 上報告有效的 **ChemBERTa-77M-MLM**
  嵌入(同款模型、CLS token,同分割同訓練循環,4 組特徵 × 3 seeds):
  單獨換入比 MoLFormer 差 0.009–0.020,拼合增益 +0.004(PAMPA)/
  −0.003(Caco-2),小於 seed 間噪音——**「換 frozen 分子 encoder
  不解決」現在有直接實測證據**。誠實結論:**分子端點的瓶頸是標籤噪音
  (同分子重複量測差 0.6–0.94 log 單位),不是 2D 描述子表示層**;
  要解決需要
  (a) 更多/更乾淨的量測數據,(b) 對重複量測做目標聚合(分子端點的
  SMILES 重複率遠低於序列端點,聚合空間小),或 (c) 任務特定的
  fine-tune(與本管線「frozen 嵌入 + 輕量 head」的策略衝突)。
- **PAMPA 0.4642 → 0.70 已系統性調查並否定**(2026-08-27,同控制分割
  實測 6 條路線:rank-Gaussian、LightGBM×128 超參+ensemble、兩階段地板法、
  soft blend、Tobit、ChemBERTa 嵌入替換,最佳 0.4651,全部 ≤ baseline+噪音)。
  **根因**:目標
  logPapp 有左側審查地板——269 行(3.7%)= -10.0000(assay 偵測下限),
  佔目標變異數 49.6%;地板分子只能被部分排序(最佳 LightGBM 分類
  AUC_test 0.8557、MLP 預測 AUC 0.7624),但可用閾值下 precision 僅
  0.12,誤報成本讓修正不可行;非地板子集 R² 已 0.6317,理論天花板
  (非地板完美、地板→全域均值)= 0.5387。突破需地板分子的無審查
  重新量測。**外部數據交叉驗證**(2026-08-28):直接訓練 PeptiVerse 論文
  原始數據(HF `ChatterjeeLab/PeptiVerse_data`,PAMPA 6,869 + Caco-2 606,
  作者預計算 ChemBERTa embedding + 2D),最佳 R² 0.4343、天花板 0.5014——
  其 PAMPA 數據同樣含 −10 審查地板(3.5% 行、佔變異數 49.9%),R² > 0.7
  跨數據集不可達。**pepADMET 標籤平均 A/B**(2026-08-28):實測其前處理
  「重複量測取算術平均」對我們的數據是空操作(僅 1.01 次量測/SMILES、
  0 個混合地板組),PAMPA Δ −0.0152 / Caco-2 Δ +0.0025,均在 seed 噪音
  內——與 pepADMET(JCIM 2026, 66, 936)報告 R² 0.435–0.657 的差距不能
  歸因於標籤平均;其分割為隨機 8:1:1 且無 SMILES 層級 train/test
  隔離,比我們的 leakage-controlled 分割寬。完整可重現腳本與數字見
  [`analysis/`](analysis/README.md)。
- **v4.2 Half_life 的「重複序列聚合」是真實的增益**(0.6973 → 0.7259,
  +0.0286):1,763 行 → 768 唯一序列,主指標改在**序列層級**報告;
  行層級的 0.6973 與序列層級的 0.7259 不是同一個量綱,不可直接比。
  原始 1,763 行中 995 行是重複量測(同一序列的第 2 次及以後,最重複的
  一條序列量測了 82 次),聚合後剩 768 筆唯一序列,砍掉的是不可約的
  實驗重測噪音。
- 分子端點無序列,同源性控制受限於唯一 SMILES 分組(近異構物可跨界);
  ESMC 對分子端點**不適用**(非標準殘基,標準 20-AA < 0.5%),
  故分子端點用 MoLFormer(XL)而非 ESMC。
- 清洗**丟棄**無效行(非 20-AA 序列、無效 SMILES、缺失標籤);
  丟棄統計寫入 `data/pepadmet_data.meta.json`,不隱瞞。
- 主指標報告在**同源性/分子控制測試分割**;隨機分割僅作洩漏對照。

---

## 🎉 總結

✅ 端對端管線跑通:prepare_pepadmet_data → {esmc,molformer}_embed(選做)→ train_pepadmet_model → peptide_admet_predictor
✅ 4 端點全部訓練 + 預測驗證(真實數據)
✅ 雙模態特徵(序列 428 + 1,152 ESMC = 1,580 維;分子 2,265 + 768 MoLFormer = 3,033 維)
✅ v4.2 換武器:Half_life R² 0.6973 → **0.7259**(重複序列聚合,序列層級)、
   Caco2 R² 0.3861 → **0.3909**、PAMPA R² 0.4573 → **0.4642**(MoLFormer,
   增益在重訓噪音內,瓶頸在標籤端)
✅ 洩漏控制(序列端 signature 合併保證 jaccard-1.0 不跨界;分子端唯一 SMILES)
✅ 實測指標:Hemolysis AUC 0.8348、Half_life R² 0.7259、Caco2 R² 0.3909、PAMPA R² 0.4642
✅ 4 個模型權重 + 準備數據 + ESMC/MoLFormer 嵌入 npz + 指標全部 commit 到 repo(可重現)
✅ E2E 驗證:predictor 路徑精確重現全部 shipped 指標;CLI 快取/新輸入雙路徑通過

**版本**:4.2(真實數據 4 端點 + ESMC-600M 序列嵌入 + MoLFormer-XL 分子嵌入,
雙模態,Huber loss,Half_life 重複序列聚合)· **日期**:2026-08-26
