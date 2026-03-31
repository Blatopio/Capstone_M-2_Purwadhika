# Saudi Arabia Used Cars — Price Prediction Model
**Capstone Project Module 2 | AI Engineer**
> By Muhammad Fachreza Alghifari

---

## Project Introduction

This project builds an end-to-end machine learning pipeline to predict used car prices in Saudi Arabia using data collected from [Syarah.com](https://syarah.com) — one of the country's primary online used car marketplaces.

The model serves as an automated price recommendation tool embedded into Syarah.com's listing process. When a seller lists a car, they input specifications such as brand, model, year, mileage, and engine size — and the model suggests a fair market price, helping sellers list competitively without requiring deep market knowledge.

---

## Business Problem

Sellers on Syarah.com lack market pricing knowledge, causing them to either overprice or underprice their listings. Overpriced cars sit unsold for extended periods, while underpriced listings erode seller trust. Both outcomes reduce completed transactions — the primary driver of Syarah.com's platform revenue.

**Stakeholder:** Syarah.com  
**Problem type:** Supervised Regression  
**Target variable:** `Price` (SAR — Saudi Arabian Riyals)

---

## Dataset

| Property | Detail |
|---|---|
| Source | [Syarah.com](https://syarah.com) |
| Raw rows | 5,624 |
| Final rows (after cleaning) | 5,400 |
| Features | 9 (after engineering) |
| Target | Price (SAR) |

**Features:**

| Column | Type | Description |
|---|---|---|
| Type | Categorical | Car model name |
| Region | Categorical | Region where car is listed |
| Make | Categorical | Car brand / manufacturer |
| Gear_Type | Categorical | Manual or Automatic |
| Origin | Categorical | Saudi / Gulf Arabic / Other / Unknown |
| Options | Categorical (Ordinal) | Standard / Semi Full / Full |
| Engine_Size | Numerical | Engine displacement in litres |
| Mileage | Numerical | Distance covered in km |
| Car_Age | Numerical (engineered) | 2024 minus manufacturing year |

---

## Methodology

### Data Cleaning
- Negotiable listings (Price = 0) filled using median price of same Make + Type + Year combination — preserving 32% of data that would otherwise be lost
- Duplicate rows removed
- Price outliers filtered using 1st–99th percentile (preserves luxury cars like Land Cruiser and Lexus RX while removing data entry errors)
- Mileage outliers filtered using 1st–99th percentile (removes physically impossible values like 20,000,000 km)
- Cars manufactured before 2000 removed (< 2% of data, not representative of current market)

### Feature Engineering
- `Car_Age` created from `Year` column — age is a more direct representation of depreciation than raw year
- Rare car types (< 10 occurrences) grouped into `Other` category — reduces unique values from 320 to 81
- `Options` encoded as ordinal (Standard=0, Semi Full=1, Full=2)
- All categorical features encoded using CatBoost Encoder (target encoding)

### Modeling
Two gradient boosting algorithms benchmarked:

| Model | RMSE | MAE | MAPE |
|---|---|---|---|
| XGBoost (benchmark) | 23,888 | 14,731 | 35.6% |
| CatBoost (benchmark) | 23,721 | 14,585 | 36.4% |
| **CatBoost (tuned)** | **23,338** | **14,317** | **35.6%** |

**Final model:** CatBoost with hyperparameter tuning via RandomizedSearchCV (20 iterations, 5-fold cross validation)

**Best parameters:**
```
iterations:    500
depth:         6
learning_rate: 0.05
l2_leaf_reg:   3
```

### Evaluation Metrics
- **RMSE** — penalizes large errors more heavily, sensitive to extreme mispredictions
- **MAE** — average error in SAR, directly interpretable for business stakeholders
- **MAPE** — percentage error, useful for comparing across different price ranges

---

## Results

The final CatBoost model predicts used car prices with an average error of **SAR 14,317 (MAE)** — meaning if a car's actual market value is SAR 60,000, the model typically predicts between SAR 46,000 and SAR 74,000.

### Top 10 Feature Importances

| Feature | Importance |
|---|---|
| Car_Age | 30.6% |
| Type | 21.0% |
| Make | 17.0% |
| Engine_Size | 14.4% |
| Mileage | 6.6% |
| Options | 3.8% |
| Origin | 2.5% |
| Region | 2.2% |
| Gear_Type | 1.9% |

**Key insight:** Car age is the strongest single predictor of price — accounting for nearly 31% of the model's decision. The car's specific model and brand together contribute another 38%, confirming that depreciation rate varies significantly by make and type.

---

## Model Limitations

- Model is trained exclusively on Syarah.com historical data and reflects Saudi Arabian market conditions at the time of data collection
- **Electric vehicles** are not present in this dataset — the model cannot reliably price EVs
- External factors such as fuel price fluctuations, government import regulations, and macroeconomic conditions are not captured
- Approximately 32% of training prices were estimated via median imputation for negotiable listings — performance on cars typically listed as negotiable may be slightly less accurate
- Physical car condition (accident history, interior quality, service records) is not available as a feature

---

## Recommendations

1. **Retrain every 6–12 months** on fresh listing data to keep Car_Age and market prices current
2. **Add EV data** as electric vehicles enter the Saudi market — EV depreciation curves differ significantly from combustion engines
3. **Incorporate external signals** — fuel price index, import tariff changes, and new car sales volume as additional features
4. **Output a price range** instead of a single prediction to better serve negotiable listings
5. **Add condition rating** as a seller-reported input at listing time — physical condition is one of the strongest real-world price drivers not currently in the model

---

## Repository Structure

```
├── README.md
├── CAPSTONE_MODUL_2_SAUDI_ARABIAN_USED_CAR.ipynb
├── model_saudi_used_cars.sav
└── data_saudi_used_cars.csv
```

---

## How to Run

1. Clone this repository
2. Open `CAPSTONE_MODUL_2_SAUDI_ARABIAN_USED_CAR.ipynb` in Google Colab or VS Code
3. Upload `data_saudi_used_cars.csv` to your working directory
4. Run all cells in order
5. The trained model will be saved as `model_saudi_used_cars.sav`

To load the saved model:
```python
import pickle
model = pickle.load(open('model_saudi_used_cars.sav', 'rb'))
model.predict(X_test)
```

---

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
catboost
category_encoders
matplotlib
seaborn
```

Install all dependencies:
```
pip install pandas numpy scikit-learn xgboost catboost category_encoders matplotlib seaborn
```

---

*Dataset source: Syarah.com — Saudi Arabia Used Cars*
