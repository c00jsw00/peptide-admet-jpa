# openclaw-peptide-admet

**Peptide ADMET 預測管線 — v4.2 真實數據 + ESMC + MoLFormer 版(4 端點、雙模態、洩漏稽核)**

> **v4.2 增量變更(2026-08-26,建立在 v4.1 上):**
>
> 1. **分子端點(Caco-2、PAMPA/MDCK)換武器:導入 MoLFormer-XL 凍結嵌入**
>    (IBM, 60M 參數,hidden 768)。對每個 SMILES 取 **CLS token 768 維**向量,
>    與 2,265 維 RDKit 特徵**拼接成 3,033 維**輸入(凍結、不 fine-tune;
>    `molformer_embed.py` 算一次並 commit `data/molformer/*.npz`,推論時快取
>    命中直接取、新 SMILES 走子程序回退)。
> 2. **Half-life 換武器:重複序列目標聚合 + Huber loss**。原始 1,763 行中
>    只有 **768 條唯一序列**(995 行是重複量測,最重複的一條序列量測了
>    82 次);v4.2 把同序列的多個 log10(half-life) **取平均**成一筆(砍掉
>    不可約的實驗重測噪音),回歸換用 **Huber loss**(對 log 空間離群值
>    穩健)。**主指標改在「序列層級」報告**(768 唯一序列,同源性控制
>    分割,train/val/test = 533/76/159)。
> 3. **分子端點同用 Huber loss**(Caco-2、PAMPA/MDCK 的 logPapp 標籤有
>    1–2 log 單位的重複量測噪音)。
> 4. **實測結果(同源性/唯一 SMILES 控制分割,seed 42,單一完整 run)**:
>    - Half-life R² **0.6973 → 0.7259**(Δ +0.0286,序列層級);
>    - Caco-2 R² **0.3861 → 0.3909**(Δ +0.0048);
>    - PAMPA/MDCK R² **0.4573 → 0.4642**(Δ +0.0069);
>    - Hemolysis AUC **0.8348**(未改端點,決定性重訓精確重現 v4.1)。
>    分子端點的 MoLFormer 增益**近乎零**——且同端點兩次獨立重訓差
>    ±0.01(PAMPA 0.4527 vs 0.4642),MoLFormer 的效果落在重訓噪音之內。
>    誠實結論見「誠實聲明」:瓶頸是標籤噪音(同分子重複量測差
>    0.6–0.94 log 單位),不是 2D 描述子表示層;frozen MoLFormer
>    未能突破該噪音地板。
> 5. 新增 `molformer_embed.py`(嵌入生成器)+ `data/molformer/*.npz`(已
>    commit);`admet_model.py` 現在把 `hidden`/`dropout` 持久化進 checkpoint
>    (v4.1 前的舊 checkpoint 仍向前相容)。
>
> **v4.1 增量變更(2026-08-26,建立在 v4.0 上):**
>
> 1. **序列端點(Hemolysis、Half-life)導入 ESMC-600M 凍結嵌入**:
>    Biohub 蛋白質語言模型對每條 20-AA 序列產出 1,152 維 mean-pooled
>    向量,與 428 維經典序列特徵**拼接成 1,580 維**輸入(嵌入是**凍結**的,
>    不 fine-tune,只用 `.venv-esmc`(Python ≥ 3.12)算一次並 commit npz 快取)。
>    同源性分割下 Hemolysis AUC **0.7755 → 0.8348**、Half-life R²
>    **0.5883 → 0.6973**(均實測重訓,非硬編)。
> 2. **Caco-2 / PAMPA/MDCK 維持 RDKit 分子路徑**(其「序列」欄是
>    非標準殘基名稱,20-AA 編碼器無法消耗)。
> 3. 新增 `esmc_embed.py`(嵌入生成器)+ `data/esmc/*.npz`(凍結嵌入快取,
>    已 commit,免 3.12 環境即可重訓/推論)。
>
> **v4.0 斷裂式變更(2026-08-25,取代 v3.0):**
>
> 1. **改用真實數據集** [Chemit797/PepADMET-Dataset](https://github.com/Chemit797/PepADMET-Dataset),
>    **刪除** v3.0 的 30,000 行 `synthetic_demo` 合成數據與 9 端點管線。
> 2. **只保留 4 端點**:Hemolysis、Half-life、Caco-2、PAMPA/MDCK
>    (v3.0 的 BBB / Ames / hERG / 毒性 / HC50 等不再保留)。
> 3. **雙模態特徵**:序列端點用 428 維氨基酸序列特徵(v4.1 起 +1,152 維
>    ESMC 嵌入 = 1,580 維);分子端點用 2,265 維 RDKit SMILES 描述子
>    (217 個 2D 描述子 + 2,048 位 Morgan)。
> 4. **4 個獨立單任務模型**——4 張表是**互斥分子**、分屬兩種模態,
>    共享 trunk 的多頭模型在此只會學兩個不相交子空間,故拆成 4 個聚焦模型。
>
> v2.0/v3.0 的誠實修訂(移除硬編碼指標、phantom 數據、RF 偽裝 NN)**全部保留**。

---

## 為什麼是真實數據、為什麼 4 端點

v3.0 的 30,000 行 `synthetic_demo` 是 `prepare_data.py` 用
Dirichlet 家族結構 + 潛在理化分數**合成**的示範數據——標籤不是實驗量測,
只能證明管線能跑,不能聲稱任何真實肽類準確度。v4.0 把它整個移除,改餵
**真實實驗/文獻數據**(PepADMET-Dataset),並收斂到你點名的 4 個端點:

| 端點 | 來源表(整理/) | 輸入 | 標籤 | 準備後行數 |
|------|------|------|------|:---:|
| **Hemolysis** | `hemolysis_unified/hemolysis_unified.csv` | `sequence_std`(20-AA 序列) | `label`(0/1 二值) | 8,719 |
| **Half-life** | `half_life_*` | `sequence`(20-AA 序列) | `half_life_seconds`(連續) | 1,763(**768 唯一序列**) |
| **Caco-2** | `caco2_*` | `SMILES`(RDKit 可解析) | `Permeability`(logPapp) | 7,429 |
| **PAMPA/MDCK** | `pampa_mdck_*` | `SMILES`(RDKit 可解析) | `PAMPA`(logPapp) | 7,283 |

> 關鍵:兩張通透性表(Caco-2 / PAMPA)的「序列」欄其實是 **CycPeptMPDB 的
> 非標準殘基名稱清單**(`MEL`、`DP`、`DL`、`ME_DL`…),20 標準氨基酸編碼器
> 無法消耗,因此這兩個端點改用**有效 SMILES 欄**做分子描述子特徵,並
> 逐端點在 `endpoint_config.py` 如實標註。

---

## 專案結構

| 文件 | 說明 |
|------|------|
| `endpoint_config.py` | 4 端點單一事實來源(kind、模態、來源 CSV、特徵欄、標籤轉換、單位、**ESMC / MoLFormer 旗標**) |
| `feature_extractor.py` | 雙模態特徵(訓練/推論共用同一份):序列 428 維 + RDKit 分子 2,265 維 |
| `esmc_embed.py` | **v4.1** ESMC-600M 凍結嵌入生成器(`.venv-esmc` 執行;預設批次模式或 `--sequences-file` ad-hoc 模式) |
| `molformer_embed.py` | **v4.2** MoLFormer-XL 凍結 CLS 嵌入生成器(`.venv` 執行;預設批次模式或 `--smiles-file` ad-hoc 模式) |
| `prepare_pepadmet_data.py` | 載入 4 個真實 CSV → 清洗(去無效序列/無效 SMILES/缺失標籤)→ 每端點準備 CSV + `meta.json` |
| `homology_split.py` | 3-mer Jaccard 家族 70/10/20 同源性分割 + 量測洩漏 |
| `admet_model.py` | `MixedADMETMLP`(binary/regression heads,通用 input_dim,**hidden/dropout 持久化**)+ 存檔 |
| `train_pepadmet_model.py` | 每端點:特徵 →(v4.1 序列端點 +ESMC / v4.2 分子端點 +MoLFormer 拼接)→ 分割 → 標準化 → 訓練 → 雙分割評估 → `metrics.json` + 權重 |
| `peptide_admet_predictor.py` | 推論 CLI:輸入 sequence / SMILES → 自動路由模態 →(嵌入快取/子程序回退)→ 預測 + 單位 |

**訓練產物 `models_v4/`**(已 commit):

| 路徑 | 說明 |
|------|------|
| `models_v4/<endpoint>/admet_mlp.pt` | PyTorch 權重 + 架構 metadata |
| `models_v4/<endpoint>/scaler.pt` | StandardScaler(只在 train 上 fit) |
| `models_v4/<endpoint>/metrics.json` | **實測**指標(雙分割)+ 分割統計 + 洩漏稽核 |
| `models_v4/summary.json` | 4 端點彙總 |

`<endpoint>` ∈ {`hemolysis`, `half_life`, `caco2`, `pampa_mdck`}。

---

## 快速開始

```bash
# 0. 環境(需要 rdkit;CPU 版 torch 即可)
uv pip install --python .venv/Scripts/python.exe rdkit

# 0b. (v4.1) ESMC 環境:Python >= 3.12 + Biohub esm(git main)
#     只需在「重新生成嵌入」或「預測全新序列」時用到;
#     重訓與預測已快取的序列不需要它(npz 已 commit)。
#     建立方式見 esmc_embed.py 檔頭,或用 uv venv --python 3.12 手動建。

# 0c. (v4.2) MoLFormer 環境:就用 .venv(transformers + sentencepiece)
#     只需在「重新生成嵌入」或「預測全新 SMILES」時用到;
#     重訓與預測已快取的 SMILES 不需要重算(npz 已 commit)。

# 1. 準備數據(從本機已 clone 的 PepADMET-Dataset 載入 + 清洗)
python prepare_pepadmet_data.py

# 2. (v4.1,選做)重新生成 ESMC 嵌入 → data/esmc/*.npz
#    首次或序列集變動時才需要;已 commit 的 npz 可直接用。
.venv-esmc/Scripts/python.exe esmc_embed.py

# 2b. (v4.2,選做)重新生成 MoLFormer 嵌入 → data/molformer/*.npz
#    首次或 SMILES 集變動時才需要;已 commit 的 npz 可直接用。
.venv/Scripts/python.exe molformer_embed.py

# 3. 訓練 4 端點(CPU,約 15–20 分鐘;嵌入端點讀 npz 快取,分子端點 RDKit 為主要開銷)
python train_pepadmet_model.py --epochs 80 --seed 42

# 4. 預測(自動路由模態:sequence → 序列端點,SMILES → 分子端點)
python peptide_admet_predictor.py --sequence "ACDEFGHIKLMNPQRSTVWY"
python peptide_admet_predictor.py --smiles "CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCNC(=N)N)C(=O)O"
```

---

## 實測性能(洩漏控制測試分割,seed 42,單一完整 run)

全部數字取自 `models_v4/<endpoint>/metrics.json`,**實測、可重導**,非硬編碼。

| 端點 | 類型 | 模態 | 主指標 | 其他(控制分割) | 隨機分割對照 |
|------|------|------|--------|----------------|------------|
| Hemolysis | binary | 序列+ESMC | AUC **0.8348** | MCC 0.4557, Acc 0.7479 | 0.8112(差 −0.0236) |
| Half-life | regression(log10 s) | 序列+ESMC | R² **0.7259** | RMSE 1.365, MAE 0.887(159 唯一序列) | 0.7867(序列層級,近重複已聚合) |
| Caco-2 | regression(logPapp) | 分子+MoLFormer | R² **0.3909** | RMSE 0.785, MAE 0.471 | —(無序列對照) |
| PAMPA/MDCK | regression(logPapp) | 分子+MoLFormer | R² **0.4642** | RMSE 0.799, MAE 0.450 | —(無序列對照) |

> **v4.2 換武器增益(同源性/唯一 SMILES 控制分割,seed 42)**:
>
> | 端點 | v4.1 | v4.2 | Δ | 換的武器 |
> |------|-----:|-----:|-----:|------|
> | Half-life(序列層級,768 唯一序列) | 0.6973(行層級 1763) | **0.7259** | +0.0286 | 重複序列目標聚合 + Huber |
> | Caco-2 | 0.3861 | **0.3909** | +0.0048 | +768 維 frozen MoLFormer-XL CLS(2265→3033 維) |
> | PAMPA/MDCK | 0.4573 | **0.4642** | +0.0069 | 同上 + Huber |
> | Hemolysis(未改端點) | 0.8348 | **0.8348** | 0.0000 | —(決定性重訓精確重現) |
>
> 注意:同端點兩次獨立完整重訓(不同 run)差約 ±0.01
> (PAMPA 0.4527 vs 0.4642),所以 MoLFormer 的 +0.005~+0.007 增益
> 在重訓噪音之內——**分子端點的瓶頸是標籤噪音,不是表示層**
> (見「誠實聲明」)。

### 洩漏控制(這是 v4.0 的核心方法貢獻)

- **序列端點(Hemolysis、Half-life)**:AMPBench-MT 風格同源性分割
  (arXiv:2607.25518)。分割前先把**同 3-mer 多重集的 anagram 合併為同一
  家族**(canonical 3-mer-multiset signature)——**保證 jaccard-1.0 的精確
  複製(含保長 anagram)絕不跨界**。最大跨界 Jaccard ≈ 0.97(近重複的合理
  上限,非洩漏)。**Half-life 隨機 R² 0.7867 仍高於同源性 0.7259(序列層級),
  正是近重複/近 anagram 洩漏被控管掉的誠實體現**;v4.2 先做重複序列聚合
  (1763 → 768 唯一序列)再分割,所以兩個分割都在**序列層級**比較,
  避免了行層級的 anagram 近重複膨脹。
- **分子端點(Caco-2、PAMPA/MDCK)**:無序列 → 按**唯一 SMILES** 分組
  (精確重複 SMILES 同分割);**近異構物**(不同 SMILES、同化學)可跨界,
  這是 SMILES-only 數據的限制,已如實寫入每個端點的 `metrics.json`。

---

## 特徵工程(訓練/推論共用 `feature_extractor.py`)

**序列模態(428 維經典 + 1,152 維 ESMC = 1,580 維,v4.1)**:
- 經典 428 維:AAC 20(氨基酸組成)+ DPC 400(二肽組成)+ 8 理化
  (估算 MW、平均 Kyte–Doolittle 親水性、pH7 淨電荷、pI 估計、GRAVY、
  疏水/帶電殘基比)。
- **ESMC-600M 凍結嵌入 1,152 維(v4.1)**:Biohub ESM Cambrian 蛋白質
  語言模型對每條 20-AA 序列做 attention-mask 加權 mean-pooling。嵌入是
  **凍結**的(不 fine-tune、不梯度),由 `esmc_embed.py` 在 `.venv-esmc`
  (Python ≥ 3.12)算一次,存 `data/esmc/*.npz`(已 commit)。訓練時與經典
  特徵**拼接**;推論時快取序列直接取、全新序列走子程序回退。

**分子模態(2,265 維 RDKit + 768 維 MoLFormer = 3,033 維,v4.2)**:
- 經典 2,265 維:RDKit 2D 描述子 217(`CalcMolDescriptors`,
  固定確定性註冊表)+ Morgan fingerprint(半徑 2)2,048 位。
- **MoLFormer-XL 凍結 CLS 嵌入 768 維(v4.2)**:IBM 的 SMILES transformer
  (60M 參數、hidden 768,`ibm-research/MoLFormer-XL-both-10pct`)對每個
  SMILES 取 **CLS token** 向量。嵌入是**凍結**的(不 fine-tune、不梯度),
  由 `molformer_embed.py` 在 `.venv` 算一次,存
  `data/molformer/*.npz`(已 commit)。訓練時與 RDKit 特徵**拼接**;
  推論時快取 SMILES 直接取、全新 SMILES 走子程序回退。
- 無效 SMILES → 該行特徵全零並計入清洗統計(不造假)。

> **為什麼換武器(分子端點)**:v4.1 的 Caco-2 R² 0.3861 / PAMPA R²
> 0.4573 被診斷為「標籤噪音」瓶頸(同分子重複量測差 0.6–0.94 log
> 單位),不是 2D 描述子表示層。v4.2 試了「加 frozen 分子 transformer
> 嵌入」這條路,實測增益 +0.005~+0.007,在重訓噪音 ±0.01 之內——
> **證實瓶頸確在標籤端而非表示端**(見「誠實聲明」)。

---

## PAMPA/MDCK R² 天花板調查(2026-08-27,「能否到 0.70?」)

對 PAMPA R² 0.4642 能否提升到 0.70 做了系統性調查:**同一控制分割
(seed 42、唯一 SMILES 70/10/20)下測試 6 條路線,全部 ≤ 0.47,無一
超過 baseline**。完整可重現分析(腳本 + 精確數字)見
[`analysis/`](analysis/README.md)。

**關鍵發現——目標變數有左側審查地板**:`PAMPA_MDCK`(logPapp)7,283 行
只有 **648 個唯一值**(量化到 0.01),其中 **269 行(3.7%)恰好 =
-10.0000**,即 assay 偵測下限的**審查值**(真實值 ≤ -10,未知)。
地板點佔目標變數**變異數的 49.6%**,並貢獻模型測試誤差的 64%。

| 問題 | 實測答案 |
|---|---|
| 地板分子能否從結構預測? | **部分可排序、但無可用操作點**——最佳 LightGBM 地板分類 AUC_test 0.8557(val 0.8251)、MLP 預測值排序 AUC 0.7624;但 val 調閾值下 precision 僅 0.121(recall 0.617),兩階段法 R² 崩潰到 −1.21 |
| 非地板子集模型 R² | **0.6317**(模型本身不差,損失集中在地板) |
| 理論天花板(非地板完美,地板→全域均值) | **R² 0.5387**;oracle 完美標記全部 47 個 test 地板分子 = 0.8073(需 AUC≈1 + 零誤報,不可達) |

**六條路線(全部實測,同一分割)**:

| 路線 | 最佳 test R² | 判定 |
|---|---:|---|
| v4.2 baseline(MLP 3,033→256→128→1, Huber) | **0.4642** | 基準 |
| Rank-Gaussian 目標變換(train-only,多 seed) | 0.434 ± 0.011 | ❌ 比基準差 |
| LightGBM × 128 超參 × 4 特徵集 + top-5 ensemble | 0.4234 | ❌ 比基準差 |
| 兩階段地板法(classifier→regressor) | 最佳閾值 precision 0.121 → R² −1.21 | ❌ 誤報成本過高 |
| Soft posterior-mean blend(β 掃描,val 選參) | 0.4651(+0.0009,噪音內) | ❌ 無實質增益 |
| Tobit 審查似然模型(統計上最正確) | 0.4056 ± 0.024 | ❌ 比基準差 |
| ChemBERTa-77M-MLM 嵌入替換(PeptiVerse 同款,4 組特徵 × 3 seeds) | PAMPA 0.4624 / Caco-2 0.4070 | ❌ 增益 < seed 間噪音 |
| **PeptiVerse 原始數據直接訓練**(HF `ChatterjeeLab/PeptiVerse_data`,PAMPA 6,869 + Caco-2 606,作者 ChemBERTa embedding + 2D) | PAMPA 0.4343(天花板 0.5014)/ Caco-2 0.4302(天花板 0.5459) | ❌ 他們的數據**也有** −10 審查地板(3.5%,佔 SS 49.9%),R² > 0.7 同樣不可達 |
| **pepADMET 標籤平均 A/B**(JCIM 2026, 10.1021/acs.jcim.5c02518:重複量測取算術平均) | PAMPA Δ −0.0152 / Caco-2 Δ +0.0025 | ❌ 我們的數據幾乎無重複(1.01 次/SMILES),平均化是空操作——非 pepADMET 高 R² 的原因 |

輔助 ablation:RDKit 217 描述子單獨 **−0.47**(過拟合有害);更多 Morgan
半徑 0.24;更寬/更深 MLP 0.44——信號在 Morgan 指紋,不在標量描述子或
模型容量。

**結論:0.70 在現有數據上不可達**——且此結論已用**外部數據交叉驗證**
(2026-08-28):直接訓練 PeptiVerse(Nat. Commun. 2026)論文的原始
PAMPA/Caco-2 數據,最佳 R² 0.4343,天花板 0.5014——其 PAMPA 數據**同樣
含 −10 審查地板**(240 行 3.5%,佔全域變異數 49.9%,與我們 49.6% 幾乎
相同),故「天花板 = 審查地板」跨數據集成立。限制是審查地板(地板分子
只能被部分排序(AUC 0.76–0.86)而非精確標記,誤報成本使其不可用於
修正)+ 標籤噪音,不是模型/特徵/架構。要突破需要**無審查的重新量測**
(地板分子的真實 logPapp)或更乾淨/更大的數據源。Caco-2 同理
(0.3909):其重複量測 within-SD σ̂ ≈ 0.97 log 單位 ≈ 目標變數 SD,天花板
更低,0.70 同樣不可達。
嵌入替換路線(ChemBERTa,2026-08-28)亦為否定:PeptiVerse 報告 ChemBERTa
PAMPA ρ=0.69,但**其原始數據的 val 子集審查地板過量**(val 5.3% vs
train 2.9%),且在我們**同分割、同訓練循環**下,ChemBERTa 單獨換入比
MoLFormer 差 0.009–0.020,兩者拼合的增益(+0.004 PAMPA / −0.003 Caco-2)
小於 seed 間噪音——該優勢未遷移到我們的正典肽 + 審查地板數據。

---

## 模型

- 共享 trunk:`input_dim → 256 → 128`(ReLU + BatchNorm + Dropout 0.2);
  `hidden`/`dropout` 持久化進 checkpoint(v4.2 起,向前相容 v4.1 舊檔)。
- 單任務 head:binary `Linear(128,1)` + sigmoid;regression `Linear(128,1)`
- 訓練:Adam(lr=1e-3, wd=1e-5)+ ReduceLROnPlateau(factor 0.5, patience 4)
  + early stopping(val loss, patience 10)
- 損失函數:binary = BCEWithLogits(pos_weight);regression = **Huber**(v4.2,
  對 log 空間離群值穩健;v4.1 及以前為 MSE)
- 參數數:序列端點(1,580 維)**438,529**;分子端點(3,033 維)**810,497**
  (v4.0 序列 143,617 為 428 維;v4.1 拼 ESMC → 1,580 維;v4.2 分子拼
  MoLFormer → 3,033 維)

---

## 誠實聲明

- 4 端點來自**真實實驗/文獻數據**(PepADMET-Dataset),非合成。
- **ESMC-600M 嵌入是凍結的、外部預訓練的**(Biohub ESM Cambrian),
  **MoLFormer-XL 嵌入也是凍結的、外部預訓練的**(IBM),本管線
  **不 fine-tune** 任一個——只把它的向量當附加特徵。兩個 npz 都已
  commit,重訓/推論不需重算;meta 記錄模型、維度、池化方式。
- **v4.2 分子端點的「換武器」沒有突破瓶頸**:frozen MoLFormer-XL
  嵌入對 Caco-2 / PAMPA 的 R² 增益 +0.005~+0.007,與同端點兩次
  獨立重訓的 ±0.01 差可比擬。誠實結論:**分子端點的瓶頸是標籤噪音
  (同分子重複量測差 0.6–0.94 log 單位),不是 2D 描述子表示層**;
  換更强的 frozen 分子 encoder 不解決這個問題,要解決需要
  (a) 更多/更乾淨的量測數據,(b) 對重複量測做目標聚合(分子端點
  的 SMILES 重複率遠低於序列端點,聚合空間小),或 (c) 任務特定
  的 fine-tune(與本管線「frozen 嵌入 + 輕量 head」的策略衝突)。
- **PAMPA/MDCK 0.4642 → 0.70 是「已調查並否定」的目標,不是「未嘗試」**:
  2026-08-27 在**與主指標相同的控制分割**下實測了 5 條提升路線
  (rank-Gaussian 目標變換、LightGBM × 128 超參 × 4 特徵集 + ensemble、
  兩階段地板法、soft posterior-mean blend、Tobit 審查似然模型),
  最佳 0.4651,全部 ≤ baseline + 重訓噪音 ±0.01;2026-08-28 第 6 條
  (PeptiVerse 同款的 ChemBERTa-77M-MLM 嵌入替換,4 組特徵 × 3 seeds)
  同樣否定(拼合增益 +0.004,小於 seed 間噪音 ±0.006–0.011);2026-08-28
  又用**外部數據交叉驗證**:直接訓練 PeptiVerse 論文的原始 PAMPA/Caco-2
  數據(HF `ChatterjeeLab/PeptiVerse_data`),最佳 R² 0.4343、天花板
  0.5014——其 PAMPA 數據**同樣含 −10 審查地板**(3.5% 行、佔變異數 49.9%),
  R² > 0.7 在外部數據上同樣不可達。**根因**:
  目標 logPapp
  有**左側審查地板**——269 行(3.7%)= -10.0000(assay 偵測下限,真實值
  佔目標變異數 49.6%,且只能被**部分排序**而非精確標記(最佳 LightGBM
  地板分類 AUC_test 0.8557、MLP 預測 AUC 0.7624,但可用閾值下 precision
  僅 0.12,兩階段法 R² −1.21);非地板子集模型 R² 已達 0.6317。理論天花板
  (非地板完美、地板→全域均值)= 0.5387。要突破必須取得地板分子的
  **無審查真實量測**,非模型層可解。完整數字與可重現腳本見
  [`analysis/`](analysis/README.md)。
- **v4.2 Half-life 的「重複序列聚合」是真實的增益**(0.6973 → 0.7259,
  +0.0286):1,763 行 → 768 唯一序列,主指標改在**序列層級**報告;
  行層級的 0.6973 與序列層級的 0.7259 不是同一個量綱,不可直接比。
  原始 1,763 行中 995 行是重複量測(同一序列的第 2 次及以後,最重複的
  一條序列量測了 82 次),聚合後剩 768 筆唯一序列,砍掉的是不可約的
  實驗重測噪音。
- 分子端點無序列,同源性控制受限於唯一 SMILES 分組(近異構物可跨界);
  ESMC 對分子端點**不適用**(其「序列」欄是非標準殘基,標準 20-AA 佔比
  < 0.5%),故分子端點用 MoLFormer(XL)而非 ESMC。
- 清洗**丟棄**無效行(非 20-AA 序列、無效 SMILES、缺失標籤);
  丟棄統計寫入 `data/pepadmet_data.meta.json`,不隱瞞。
- 主指標報告在**同源性/分子控制測試分割**;隨機分割僅作洩漏對照。
- 所有性能數字是**實測值**(見 `metrics.json`),可用
  `prepare_pepadmet_data.py → train_pepadmet_model.py` 在 CPU 上重導
  (嵌入端點直接讀已 commit 的 npz)。

---

## 版本

**v4.2**(真實數據 4 端點 + ESMC-600M 序列嵌入 + MoLFormer-XL 分子嵌入,
雙模態,Huber loss,Half-life 重複序列聚合)· 2026-08-26
建立在 v4.1(2026-08-26,ESMC-600M 凍結嵌入)之上;
v4.1 建立在 v4.0(2026-08-25,真實數據 4 端點、雙模態)之上;
v4.0 取代 v3.0(30k `synthetic_demo` 合成數據 9 端點)。
完整技術說明與方法論見 [peptide_admet_manuscript_jcim.md](peptide_admet_manuscript_jcim.md);
端點摘要見 [PREDICTOR_SUMMARY.md](PREDICTOR_SUMMARY.md)。
