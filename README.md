# Forecasting Time-Varying Cross-Sectional Regression Coefficients with Machine Learning: Evidence from Taiwan

RC forecasting 補跑實驗的全部輸出與程式碼(還原報酬版,2026-10-12)。

所有表格、圖與 audit CSV 都由 `code/` 內同一套程式、同一份預測面板與同一份月報酬序列產生。

## 資料夾結構

```
.
├── README.md      本文件(說明)
├── tables/        正文與附錄的表:Table 2–7、Appendix A.1–D.1、無風險利率與 TAIEX 月資料
├── figures/       Figure 1(六張)、Figure 2(四張)與對應的資料 CSV
├── audit/         稽核用逐月資料 (9-1)(9-2)(9-3)(9-5)、還原報酬與超參數稽核
└── code/          完整程式碼,run_all.sh 為執行順序
```

(9-4) `monthly_stock_scores_and_holdings.csv` 約 217 萬列、770 MB,超過 GitHub 單檔上限,以壓縮檔放在本 repo 的 [Releases](../../releases) 頁面。pandas 可直接讀 `.csv.gz`。

## 目錄

- [一、這一版改了什麼](#一這一版改了什麼)
- [二、資料來源與還原方法](#二資料來源與還原方法)
- [三、仍與正文不同、需要決定的地方](#三仍與正文不同需要決定的地方)
- [四、結果和上一版不同](#四結果和上一版不同)
- [五、實作上的定義](#五實作上的定義)
- [六、檔案對照](#六檔案對照)
- [七、如何重跑](#七如何重跑)

## 一、這一版改了什麼

上一版用的是未還原月底收盤價、只有 964 檔存活公司(其中 333 檔其實是上櫃股)。這一版依正文 2.1、2.2 節重建資料:

| 正文寫法 | 這一版的做法 |
|---|---|
| split- and distribution-adjusted prices | 以證交所每日行情與官方參考價計算除權息、減資、面額變更調整後的報酬 |
| 臺灣證券交易所上市普通股 | 只取上市普通股,不含上櫃、ETF、TDR、特別股 |
| 含下市、合併、停止交易股票 | 納入證交所終止上市名單內的公司 |
| 樣本 2000/01–2024/12 | 報酬自 2000/01 起算 |
| 歷史報酬不足 12 個月者剔除 | 已套用 |
| 剔除全額交割、重整股 | 以證交所「變更交易」名單於每月底剔除,2004/12 起 |
| t−1 需有市值與成交量 | 已套用(形成月需有成交量且市值為正) |
| Table 1 訊號定義 | 改為 mom6m 5 個月、mom12m 11 個月(見第三節) |
| 兩碼產業別 | 改用證交所兩碼產業別 |
| TAIEX 與投組報酬基礎一致 | 投組報酬含股利,所以 TAIEX 一律用含息的發行量加權股價報酬指數 |

資料來源不是 TEJ,而是公開的官方資料

## 二、資料來源與還原方法

**來源**

- 每日行情(收盤價、漲跌、成交量、成交金額):證交所每日收盤行情,經 FinMind TaiwanStockPrice 取得。
- 除權息參考價:證交所「除權除息計算結果表」(TWT49U),2003/05/05 起,17,638 筆。
- 減資恢復買賣參考價:證交所 TWTAUU,2011/01/01 起,389 筆。面額變更參考價:TWTB8U,8 筆。
- 變更交易(全額交割)名單:證交所 TWT85U,2004/12/24 起,取每月最後交易日的名單,共 241 個月、158 檔股票曾列入。
- 終止上市公司:證交所終止上市公司名單。上市日期與產業別:證交所上市公司基本資料。
- 發行股數:FinMind TaiwanStockShareholding(2004/02 起),用於原始市值檔沒有的股票。原始市值檔(raw_size.xlsx)內已有的 631 檔上市股沿用該檔。
- 大盤:發行量加權股價報酬指數(含息),證交所,經 FinMind。不含息的價格指數只作輔助欄。無風險利率:中央銀行金融統計月報。

**還原方法**

- 每日還原報酬 = 當日收盤價 ÷ 參考價 − 1。一般交易日的參考價是前一日收盤價;除權息、減資、面額變更當日的參考價是證交所公布的參考價。
- 月報酬是當月每日還原報酬的連乘。
- 公司行動日的認定有兩種:出現在官方表內;或每日行情的漲跌為 0 但收盤價與前一日不同(交易所當日不與前一日比價)。
- 找不到官方參考價的公司行動日,當日報酬記為 0。這包括 2003/05/05 以前的除權息、2011 年以前的減資、以及股票還在上櫃期間的事件。這樣處理的誤差上限是單日漲跌幅,不會產生假報酬。
- 各期間的處理筆數見 audit_corporate_action_days_summary.csv:2011 年以後有官方參考價的 10,948 筆、無參考價的 467 筆;2003/05–2010 為 4,078 筆與 885 筆;2000–2003/04 全部 1,297 筆都沒有官方參考價。
- 上市前的興櫃期間報價(議價均價,不在升降單位上)已剔除。

**外部檢查** audit_adjusted_return_check_vs_yahoo.csv 把這套月報酬與 Yahoo Finance 還原收盤價的月報酬逐筆比對(Yahoo 資料沒有用在計算裡)。樣本外期間 2012/01–2024/12 共 136,745 個股票月,97.7% 相差不到 1 個百分點,等級相關 0.9955

## 三、仍與正文不同、需要決定的地方

1. **2004/11 以前沒有全額交割名單。** 證交所的變更交易名單自 2004/12/24 起才有,更早的月份沒有做這項篩選。驗證期 2007–2011 與樣本外 2012–2024 都完整涵蓋。整個樣本期間共剔除 4,446 個股票月。
2. **2000–2004 年的下市公司不完整。** 公開行情對已下市公司多半只回溯到 2004/02。因此 2005 年以後樣本含下市公司(驗證期 2007–2011 與樣本外 2012–2024 都涵蓋),但 2000–2004 年實際上只有後來存活的公司。曾進入樣本的 1,057 檔中,111 檔後來下市。
3. **下市當月的報酬。** 持有的股票若次月完全沒有成交(多為合併下市),次月報酬記為 0。樣本外 134,285 個合格股票月中有 88 個。
4. **Table 1 的時間標記。** Table 1 註解寫 r(t−k) 是形成月 t 之前 k 個月的報酬,式 (5) 又用 Z(t) 預測 r(t+1)。照字面讀,形成月當月的報酬完全沒被用到。程式的做法是 k = 1 代表形成月當月(最近一個已知報酬):mom1m = r(t);mom6m = t−4 到 t 共 5 個月;mom12m = t−10 到 t 共 11 個月;mom36m = t−35 到 t−12;chmom = R(t−5..t) − R(t−11..t−6)。
5. **TAIEX 改用含息的報酬指數。** 股票報酬現在含股利,所以所有表、圖、audit 檔的 TAIEX_return 都是發行量加權股價報酬指數的月報酬,勝率與 alpha 也以它計算。不含息的價格指數另附為輔助欄(Table 5 最後一欄、Table D.1 的 vs_price_index 欄)。正文寫 TAIEX 的地方需要說明是報酬指數。
6. **2003 年以前的 RC 序列較短。** 樣本改從 2000/01 起算後,第一個實現 RC 是 2000/12(mom36m 為 2002/12),Table 2 Panel B 的觀察數變成 288 與 264。

## 四、結果和上一版不同

資料換成還原報酬、上市全樣本後,結論方向改變:

| 項目 | 上一版(未還原、存活公司) | 這一版 |
|---|---|---|
| 樹模型 R²OOS | 四個訊號為正,2%–5% | mom1m、mom6m、mom12m、chmom 為正,0.8%–11.8%;mom36m、indmom 為負 |
| LightGBM、GBRT 的 R²OOS | LightGBM 四正二負 | 兩者都是四正二負(mom36m、indmom 為負) |
| 等權重淨夏普最高 | AR(1) 0.966、Historical mean 0.954 | LightGBM 0.821、OLS 0.766、GBRT 0.750、NN5 0.743 |
| 等權重基準模型淨夏普 | 0.59–0.97 | 0.29–0.49 |
| 市值加權淨夏普最高 | NN2 0.618 | NN5 0.707、OLS 0.681、Random Forest 0.660、NN1 0.637 |
| 市值加權基準模型淨夏普 | 0.29–0.45 | 0.00–0.45 |
| 排除最小 20% 後等權重淨夏普 | 多數降到 0.15–0.45 | ML 多數維持在 0.59–0.81 |
| Oracle 淨夏普 | 2.24–2.51 | 2.20–2.31 |

- **模型預測的投組勝過三個基準預測**,等權重與市值加權都是,排除小型股與低流動性股後仍成立,前後半期也都成立。
- **樹模型的預測準度全面高於 OLS。** 四個樹模型在六個訊號上的 R²OOS 都高於 OLS;OLS 有四個訊號為負。
- **投組績效上 ML 與 OLS 互有高低。** 等權重 LightGBM 第一、OLS 第二;市值加權 NN5 第一、OLS 第二。「非線性模型優於線性模型」在預測準度上成立,在投組績效上不能這樣寫。
- **mom36m 沒有任何模型能預測**(R²OOS 全負)。
- **alpha 都不顯著。** Table D.1 對報酬指數的 alpha,t 值最高是 OLS 等權重 1.67。三個基準的 alpha 為負。
- **與大盤比較:** 含息報酬指數夏普 0.912(不含息價格指數為 0.657)。等權重與市值加權都沒有模型超過 0.912。「打敗大盤」不能寫。
- **等權重的選定 cutoff 多為 1%**(約 7–9 檔),持股很集中。Table C.1 顯示多數 ML 模型在 5%、10% 的淨夏普更高(例如 Random Forest 10% 為 0.858)。
- **驗證期夏普偏低**(0.17–0.53),因為 2007–2011 含金融海嘯。
- **結果對樣本定義敏感。** 加入全額交割篩選前後,各模型的選定 cutoff 與排名有變動(例如等權重 LightGBM 由 0.64 變為 0.82)。模型之間的小幅差距不宜過度解讀。

## 五、實作上的定義

- **月份定義**:OOS 為 target(報酬)月 2012/01–2024/12,共 156 個月,對應 formation 月 2011/12–2024/11。驗證期為 target 月 2007/01–2011/12(60 個月)。
- **合格股票**:形成月已在證交所上市、當月有成交量、市值為正、累計至少 12 個月報酬、月底不在變更交易名單內。實現 RC 的橫斷面迴歸與投組都只用合格股票。
- **預測資訊集**(所有 ML 模型相同,共 13 個變數):該訊號自身 realized RC 的落後 1、2、3、6、12 期;過去 3、6、12 個月的移動平均與移動標準差;target 月份的 sin / cos。
- **超參數**:每個模型一組設定,以驗證期六個訊號的平均 R²(相對 expanding historical mean)最高者為準,2012/01 之前固定。候選格點與選定值見 appendix_A1_model_configurations.csv。
- **AR(1)**:每月以當時可得的全部歷史遞迴估計截距與係數。
- **投組宇集**:合格股票中六個訊號皆有值者。2006/12 為 578 檔,2011/12 為 694 檔,2024/11 為 927 檔。
- **Z 分數**:在投組宇集內對每個訊號做 1/99 縮尾後標準化。
- **產業動能**:同一證交所兩碼產業內,合格股票 mom12m 以形成月市值加權平均。已下市公司的產業別以 FinMind 產業名稱對應到證交所代碼。
- **檔數**:max(1, round(N × k%))。分數相同時依股票代號排序。
- **週轉率**:式 (13),w⁻ 為前一期投組經當月報酬漂移後的權重。驗證期第一個月前無持股,週轉率為 0.5。
- **夏普比率**:√12 × mean(r − rf) / sd(r − rf),樣本標準差。
- **Newey–West**:Bartlett kernel、6 期落後、無小樣本修正(statsmodels HAC)。
- **TAIEX**:TAIEX_return 是含息報酬指數的月報酬;TAIEX_price_index_return 是不含息價格指數,僅供參考。2012/01–2024/12 的月均報酬分別為 1.147% 與 0.845%。
- **勝率**:淨報酬高於當月 TAIEX_return 的樣本外月份比例。
- **無風險利率**:臺灣銀行一個月期定期存款固定利率(年息 %),來源為中央銀行金融統計月報「銀行業利率」(開放資料 EG2WM01)。報酬月 m 使用 m−1 月底牌告利率除以 12。
- **Break-even cost**:OOS 平均月毛報酬 ÷ 平均月週轉率,以 bps 表示。
- **規模篩選**:每個形成月排除宇集內市值最低 20%,之後重算 Z 分數、分數、排名、權重、週轉率。cutoff 百分比不變。
- **流動性篩選**:每個形成月排除當月日均成交金額最低 20%。
- **COVID 排除**:只從績效統計移除 2020/01–2021/12 的 24 個月報酬,模型估計時間線不變。

## 六、檔案對照

| 需求 | 檔案 |
|---|---|
| Table 2、Table 3 | `tables/table2_summary_statistics.csv`、`tables/table3_correlations.csv`(資料換了,這兩張表要更新) |
| [1] Figure 1 | `figures/fig_realized_rc_series_(signal).png` 六張、`figures/fig_realized_rc_series_data.csv` |
| Table 4、A.2 | `tables/table4_r2oos.csv`、`tables/appendix_A2_mae.csv` |
| [2] Table 5 | `tables/table5_portfolio_performance.csv`、`tables/table5_TAIEX_reference.csv`(含報酬指數與價格指數兩列) |
| [3] Figure 2 | `figures/fig_cumulative_net_wealth_(a/b/c/d).png`、`figures/fig_cumulative_net_wealth_data.csv`(Y 軸為對數刻度,四張相同;圖上畫的是含息報酬指數,資料檔另有價格指數欄) |
| [4] Table 6 | `tables/table6_transaction_cost_investability.csv` |
| [5] Table 7 | `tables/table7_subperiod_stability.csv` |
| [6] Table A.1 | `tables/appendix_A1_model_configurations.csv`(供整理用) |
| Table B.1、C.1 | `tables/appendix_B1_oracle.csv`、`tables/appendix_C1_net_sharpe_all_cutoffs.csv` |
| [7] Table D.1 | `tables/appendix_D1_risk_adjusted_performance.csv`、`tables/risk_free_rate_and_TAIEX_monthly.csv` |
| (9-1) | `audit/monthly_rc_forecast_panel.csv` |
| (9-2) | `audit/monthly_signal_summary.csv`(策略沒有 selected-signal 規則,dominant_signal 只是事後整理欄位) |
| (9-3) | `audit/monthly_portfolio_returns.csv`(多四欄:universe 為 base / ex_small / ex_illiquid,重現 Table 5 請篩 base;n_holdings;n_eligible;TAIEX_price_index_return) |
| (9-4) | [Releases](../../releases) 內的 `monthly_stock_scores_and_holdings.csv.gz`(約 217 萬列,解壓後 770 MB,Excel 打不開,請用 pandas 或 R) |
| (9-5) | `audit/oracle_monthly_returns.csv` |
| 還原報酬稽核 | `audit/audit_corporate_action_days_summary.csv`、`audit/audit_adjusted_return_check_vs_yahoo.csv` |
| 超參數稽核 | `audit/audit_validation_grid_*.csv` |
| 程式 | `code/`(`run_all.sh` 為執行順序) |

## 七、如何重跑

```bash
pip install -r code/requirements.txt
bash code/run_all.sh
```

- `00*` 開頭的步驟需要網路(證交所、FinMind、中央銀行開放資料)。FinMind 免費額度每小時約 300 次請求,完整下載約需 6.5 小時,程式可中斷續傳。
- 其餘步驟離線執行、結果固定(隨機種子 42),約 15 分鐘。
- 原始下載資料不放在 repo 內。程式會把資料下載到 `data/`,輸出寫到 `out/deliverables/`。
- `data/raw_size.xlsx`(631 檔上市股的月底市值)是既有檔案,不在 repo 內,重跑前需自行放入。
