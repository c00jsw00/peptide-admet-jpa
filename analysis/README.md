# PAMPA R² 天花板分析(2026-08-27)

本目錄保存 v4.2 PAMPA 端點 R² = 0.4642「為何無法達到 0.70」的完整調查。
**結論:0.70 在当前數據上不可達**——不是模型/特徵/架構問題,是目標變數的
**左側審查(left-censoring)地板 + 標籤噪音**的數學限制。以下每條結論都有
可重現的腳本與實測數字。

## 結論摘要

| 問題 | 答案(實測) | 證據腳本 |
|---|---|---|
| 目標變數結構 | 7,283 行只有 **648 個唯一值**(量化到 0.01);**269 行(3.7%)恰好 = -10.0000**,是 assay 偵測下限的**審查地板**;floor 佔全域 SS 的 **49.6%** | `r2_ceiling.py` |
| R² 損失集中在哪 | 地板點佔總平方誤差 **64%**(47 個 test 點,MAE 3.43 log 單位);非地板子集模型 R² 已達 **0.6317** | `round3_strong_2d.py` 段 A |
| 地板分子可否從結構預測? | **只能部分排序、無可用操作點**。最佳 LightGBM 地板分類 AUC_test = **0.8557**(val 0.8251)、MLP 預測值排序 AUC 0.7624;val 調閾下 precision 僅 0.121(recall 0.617),兩階段法 R² 崩潰 | `floor_predictability.py`、`soft_blend.py` |
| 理論天花板 | 完美非地板回歸 + 地板→全域均值 = **R² 0.5387**;oracle 完美識別地板 = 0.807(不可達) | `r2_ceiling.py` |
| 6 條提升路線 + 2 份外部數據 | 全部 ≤ 0.47(見下表 + 第 6/7/8 節),無一超過 baseline 0.4642 | round1–3 + blend + tobit + ChemBERTa + PeptiVerse + 標籤平均 A/B |

## 六條路線的實測結果(同一分割:seed 42、unique-SMILES 70/10/20)

| 路線 | 最佳 test R² | 判定 |
|---|---:|---|
| v4.2 baseline(MLP 3,033→256→128→1, Huber) | **0.4642** | 基準 |
| 1. Rank-Gaussian 目標變換(train-only mapping,多 seed) | 0.434 ± 0.011 | ❌ 比基準差 |
| 2. 強 2D:LightGBM × 128 超參 × 4 特徵集 + top-5 ensemble | 0.4234 | ❌ 比基準差 |
| 3. 兩階段地板法(classifier→regressor) | AUC 0.86 但最佳閾 precision 0.121 → R² −1.21 | ❌ 誤報成本使地板不可用於修正 |
| 4. Soft posterior-mean blend(β 掃描,val 選參) | 0.4651(+0.0009,噪音內) | ❌ 無實質增益 |
| 5. Tobit 審查似然(統計上最正確) | 0.4056 ± 0.024 | ❌ 比基準差 |
| 6. 嵌入替換:ChemBERTa-77M-MLM(PeptiVerse 同款) | 0.4624(PAMPA C,增益 < 噪音) | ❌ 無顯著增益,見第 6 節 |

輔助 ablation(round1):RDKit 217 描述子單獨 **−0.47**(有害,過拟合);
更多 Morgan 半徑 0.24(過拟合);更寬/更深 MLP 0.44(無幫助);信號在
Morgan 指紋,不在標量描述子或模型容量。

## 第 6 條路線:嵌入替換 — ChemBERTa(PeptiVerse 同款,2026-08-28)

PeptiVerse(Nat. Commun. 2026, s41467-026-74167-w)報告「嵌入選擇 > 模型
架構」,且在 PAMPA 上 ChemBERTa(Spearman ρ=0.69)勝過 PeptideCLM
(ρ=0.59)、Caco-2 0.80 vs 0.75。他們用的模型與我們測試的**完全相同**:
`deepchem/ChemBERTa-77M-MLM`(CLS token,384 維 frozen)。我們在**同分割、
同訓練循環**(直接 import 主訓練管線的 `train_endpoint_model`,Huber/Adam/
early-stop 逐字一致)下測試 4 組特徵 × 3 seeds(42/123/7):

| 端點 | A: mol+MoLFormer(v4.2 重跑) | B: mol+ChemBERTa | C: mol+MoLFormer+ChemBERTa | D: ChemBERTa 單獨 |
|---|---:|---:|---:|---:|
| PAMPA | 0.4581 ± 0.0063 | 0.4494 ± 0.0187 | 0.4624 ± 0.0083 | 0.3394 ± 0.0254 |
| Caco-2 | 0.4100 ± 0.0079 | 0.3896 ± 0.0035 | 0.4070 ± 0.0109 | 0.2475 ± 0.0130 |

**結論:否定**。ChemBERTa 單獨換入(B)比 MoLFormer 差 0.009–0.020;
兩者拼合(C)在 PAMPA 平均 +0.0043、Caco-2 −0.0030,**都小於 seed 間
噪音(±0.006–0.011)**,不構成可報告的增益。PeptiVerse 的優勢沒有遷移到
我們的數據——合理原因:他們的 PAMPA 來自 CycPeptMPDB(非正典/環肽)、
無審查地板、以 Spearman ρ 報告;我們是正典肽、有佔 49.6% SS 的審查
地板、以 R² 報告。管線維持 frozen MoLFormer-XL 不變,ChemBERTa 嵌入
不進入 `models_v4/`。

## 第 7 條:PeptiVerse 原始數據交叉驗證(2026-08-28)

用 PeptiVerse(Nat. Commun. 2026, s41467-026-74167-w)**論文自己的原始數據**
(HuggingFace `ChatterjeeLab/PeptiVerse_data`)直接訓練,驗證「天花板 = 審查
地板」是否跨數據集成立。PAMPA 6,869 行(CycPeptMPDB 環肽,欄位名
`sequence` 但實為 SMILES)+ Caco-2 606 行;特徵 = 作者**預計算的
ChemBERTa-77M-MLM 384d embedding**(即其報告 ρ=0.69 的那個)+ 自建
RDKit/Morgan 2D 2265d;分割 (a) 作者 train→val(5,187→1,682)對照其
ρ,(b) 我們的 unique-SMILES 70/10/20(4,808/687/1,374,無重複 SMILES
洩漏):

| 端點 | 配置 | test R² | test Spearman ρ |
|---|---|---:|---:|
| PAMPA(N=6,869;floor 240 行 3.5%,SS 佔 49.9%,天花板 **0.5014**) | E1 ChemBERTa | 0.3892 ± 0.0135 | 0.7490 ± 0.0055 |
| | E2 ChemBERTa+2D | **0.4343 ± 0.0030** | **0.7696 ± 0.0059** |
| | E3 2D only | 0.4209 ± 0.0101 | 0.7631 ± 0.0074 |
| Caco-2(N=606;floor 15 行 2.5%,SS 佔 45.4%,天花板 **0.5459**) | E2 ChemBERTa+2D | 0.4302 ± 0.0919 | 0.6025 ± 0.1188 |

**結論**:

1. **R² > 0.7 在他們的數據上同樣數學上不可達**:這份 PAMPA 數據**也有
   −10 審查地板**(240 行 3.5%,佔全域 SS 49.9%;Caco-2 15 行,45.4%),
   oracle 天花板 0.5014/0.5459——與我們數據(49.6%、0.5387)幾乎相同。
   「天花板 = 審查地板」**跨數據集成立**。
2. 最佳 R² = 0.4343(PAMPA E2),甚至**低於**我們 PepADMET 數據的 baseline
   0.4642。
3. 作者 split 的 val ρ = 0.6330(E1)/ 0.6451(E2),與其報告的 0.69 同級但
   略低;注意其 train/val **floor 比例失衡**(PAMPA val 5.3% vs train
   2.9%;Caco-2 val 7.4% vs 1.2%)——val 過量審查行,拖低 val ρ。我們
   的 leakage-controlled test(E2)ρ = 0.7696 反而更高(test floor 3.4%
   較少 tie)。
4. 非地板子集 R² = 0.6031(E2),與我們數據的 0.6317 相近——**兩份數據的
   非地板模型表現一致,差距仍全在地板**。
5. ChemBERTa+2D 比純 2D 在 PAMPA 高 0.0134(3 seed 方向一致),幅度在
   噪音邊緣;與第 6 條(我們數據上增益 < 噪音)一致:嵌入增益小且不穩。

## 第 8 條:pepADMET 標籤平均 A/B(2026-08-28)

pepADMET(J. Chem. Inf. Model. 2026, 66, 936–946, 10.1021/acs.jcim.5c02518)
的 Methods 第 (1) 步對重複量測做**標籤平均**("If the same molecule
corresponded to multiple experimental values, their arithmetic mean was
used")+ InChIKey 去重。其 PAMPA/Caco-2 端點 test R² 報告為 0.435–0.657。
A/B 實驗(同特徵、同 leakage-controlled unique-SMILES 70/10/20 分割、同
MLP+Huber 訓練循環;A = 原始行,即 v4.2 協議;B = 每唯一 SMILES 一筆、
y = 重複量測算術平均,地板行原值進平均、不做再審查):

| 端點 | 行數→唯一 SMILES | A: 原始行 | B: 標籤平均 | Δ mean |
|---|---|---:|---:|---:|
| PAMPA | 7,283 → 7,177(平均 1.015 次/SMILES) | 0.4472 ± 0.0337 | 0.4320 ± 0.0285 | **−0.0152** |
| Caco-2 | 7,429 → 7,376(平均 1.007 次/SMILES) | 0.3762 ± 0.0207 | 0.3787 ± 0.0168 | **+0.0025** |

**結論:標籤平均不是 pepADMET 高 R² 的原因**。

1. 我們的數據**幾乎沒有重複量測**:PAMPA 僅 104 個 SMILES 有 2–3 筆
   (210 行)、Caco-2 僅 52 個(105 行),平均 1.01 次/SMILES——平均化
   幾乎是空操作。
2. **0 個 SMILES 混合地板/非地板、0 個被平均抬出地板**——標籤平均甚至
   不改變審查地板的結構(與「重複組 within-SD σ̂ ≈ 0.20」的診斷一致:
   重複噪音本來就小,主要瓶頸是審查,不是重複)。
3. Δ 全在 seed 噪音內(PAMPA −0.0152 甚至為負:seed 42 B=0.4272 <
   A=0.4642,seed 123 B=0.4669 ≈ A=0.4759,seed 7 持平)。**A seed=42
   精確重現已提交的 baseline 0.4642**,協議校準正確。

因此 pepADMET 的 0.435–0.657 與我們的 0.46/0.40 的差距**不能歸因於標籤
平均**。剩餘的合理解釋:(a) 它們用**隨機 8:1:1 分割且無 SMILES/InChIKey
層級的 train/test 隔離**(Methods 亦提 0.75:0.25 兩種描述),結構近鄰可能
同時出現在 train 與 test,test R² 被推高——我們的 leakage-controlled 分割
更嚴;(b) 它們的模型是 GNN + 描述子融合 + RFE-RF 特徵選取,與我們的
MLP 不同。其「模型好壞」標準為 **R² 為主、MAE/RMSE 並列,超參以
grid-search + 交叉驗證 R² 最大化選模**(Methods 原文),未報告 AUC、未
討論 −10 審查地板。

## 為何 0.7 不可達(數學)

R² = 1 − SSE/SST。PAMPA 目標 y = logPapp 的變異數結構:

| 區間 | 行數 | 全域 SS 占比 | 說明 |
|---|---:|---:|---|
| y = -10.0000(審查地板) | 269 | **49.6%** | assay 偵測下限,真實值未知(≤ -10);行值近常數,占比全部來自偏移 |
| (-10, -6] | 5,615 | 22.6% | 模型 R² 0.63 的 bulk |
| y > -6(高滲透尾) | 1,457 | 27.8% | 重尾 |

- 地板行佔**近一半**總變異數(全域平方偏差的 49.6%)。模型只能把地板
  分子**部分排序**(AUC 0.76–0.86)、無法精確標記(最佳閾 precision 0.12),
  所以這 49.6% 的變異數大部分無法解釋 → R² 上限被鎖在 ~0.54。
- 即便 oracle 完美標記地板分子,上限 0.807 也依賴不可達的條件。
- 現行 0.4642 距離 0.5387 天花板尚有 0.074 的差距,但**橋過天花板本身
  不可能**(它要求非地板回歸完美);天花板之上的 0.70 需要地板分子有
  可預測的真實值——即**無審查的重新量測**。

## 要真正達到 0.7 需要什麼

1. **無審查的重新量測**:對 y = -10.0000 的 269 個分子做更敏感 assay
   (或文獻值),取得真實 logPapp。這直接移除 49.6% 變異數的不可預測部分。
2. **降低重複量測噪音**:重複組 within-SD σ̂ ≈ 0.20(log 單位)已不低,
   但相對 var(y)=1.26 尚可控;主要瓶頸是審查,不是重複噪音。
3. **更大樣本**:7,283 行中只有 648 個唯一值(0.01 量化),有效樣本量遠
   低於 7,283;更多**獨立量測的分子**才能壓低標籤噪音。

## 重現方法

```bash
cd <repo root>
.venv/Scripts/python.exe analysis/tobit_sanity.py          # 秒級,驗證 NLL 函數
.venv/Scripts/python.exe analysis/r2_ceiling.py            # ~2 min,天花板分解(先跑,建快取)
.venv/Scripts/python.exe analysis/round1_feature_ablation.py   # ~25 min
.venv/Scripts/python.exe analysis/round2_rank_gaussian.py      # ~25 min(首次)
.venv/Scripts/python.exe analysis/round3_strong_2d.py          # ~30 min
.venv/Scripts/python.exe analysis/floor_predictability.py      # ~10 min
.venv/Scripts/python.exe analysis/soft_blend.py                # ~10 min
.venv/Scripts/python.exe analysis/tobit_censored.py            # ~15 min

# 第 6 條路線(ChemBERTa 嵌入替換):
.venv/Scripts/python.exe chemberta_embed.py                    # ~30 min,生成 data/chemberta/*.npz(需 transformers + HF 下載)
.venv/Scripts/python.exe analysis/chemberta_retrain.py pampa   # ~2 min(讀既有快取)
.venv/Scripts/python.exe analysis/chemberta_retrain.py caco2   # ~12 min(首次建 _caco2_feat_cache.npz)

# 第 7 條(PeptiVerse 原始數據交叉驗證;data/peptiverse/*.parquet 已 commit):
.venv/Scripts/python.exe analysis/peptiverse_experiment.py both  # 首次 ~10 min(建 _pv_*_feat_cache.npz),之後 < 2 min

# 第 8 條(pepADMET 標籤平均 A/B;讀既有快取):
.venv/Scripts/python.exe analysis/label_avg_experiment.py both    # ~12 min,12 組(A/B × 2 端點 × 3 seeds)
```

- 所有腳本用**與主訓練管線逐字相同的分割**(`train_pepadmet_model.py`
  的 `split_molecular`,seed 42),所以 R² 與已提交的 0.4642 直接可比。
- 特徵快取 `_pampa_feat_cache.npz`(176MB,repo root)由 `r2_ceiling.py` /
  round2 / round3 首次運行建立(~8 min,CPU),其餘腳本直接讀取;
  `.gitignore` 已排除。
- `import common`(各腳本檔頭)會把 CWD 導回 repo root,故可從任意目錄啟動。
- 結果 log 本輪產生於:`_pampa_diag.log`、`_pampa_r2.log`、`_pampa_r3.log`、
  `_pampa_floor.log`、`_pampa_blend.log`、`_pampa_tobit.log`(均被 gitignore)。

## 各腳本內容

| 檔 | 做什麼 | 關鍵輸出 |
|---|---|---|
| `common.py` | 共用:CWD re-root、sys.path、`FEAT_CACHE` 路徑 | — |
| `r2_ceiling.py` | 天花板分解:變異數占比、非地板 R²、地板 ss 占比、天花板 R² 公式 | 0.4642 / 0.6317 / 64% / 0.5387 |
| `round1_feature_ablation.py` | 特徵/目標/架構 ablation(12 組) | 信號定位:信號在 Morgan |
| `round2_rank_gaussian.py` | Rank-Gaussian 目標變換,誠實 train-only mapping,多 seed | 0.434±0.011(否決) |
| `round3_strong_2d.py` | LightGBM 128 超參 × 4 特徵 + ensemble + 兩階段 | 最佳 0.4234;區間 R² 診斷 |
| `floor_predictability.py` | 地板分子分類器 AUC(全特徵/描述子)+ oracle 天花板 sweep | AUC_test 0.8557;oracle 0.807 |
| `soft_blend.py` | 後驗均值 soft blend,β 掃描,val 選參 | 0.4651(+0.0009) |
| `tobit_censored.py` | Tobit 審查似然模型(50 ep,多 seed) | 0.4056±0.024 |
| `tobit_sanity.py` | `_log_phi_cdf`/`nll_tobit` 數值驗證(vs scipy) | ALL PASSED |
| `chemberta_retrain.py` | 嵌入替換實驗(PeptiVerse 同款 ChemBERTa-77M-MLM),4 組特徵 × 3 seeds,`[pampa\|caco2]` | PAMPA C 0.4624 / Caco-2 C 0.4070,均 < 噪音增益(否決) |
| `peptiverse_experiment.py` | 第 7 條:PeptiVerse 原始數據(HF `ChatterjeeLab/PeptiVerse_data`)× 3 特徵 × 3 seeds,兩種分割(作者 train→val + 我們 unique-SMILES),`[pampa\|caco2\|both]` | PAMPA 最佳 R² 0.4343(天花板 0.5014)/ Caco-2 0.4302(天花板 0.5459),>0.7 不可達跨數據集成立 |
| `label_avg_experiment.py` | 第 8 條:pepADMET 標籤平均 A/B,2 端點 × A/B × 3 seeds,`[pampa\|caco2\|both]` | PAMPA Δ −0.0152 / Caco-2 Δ +0.0025(均噪音內,否決) |
