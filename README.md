# Freight Rate Prediction — Spotter ML Engineer Assessment

Predict `posted_rate` (the price in $ of a truck load) for 12,000 loads in Nov–Dec 2025, using 48,000 labelled loads from Jan–Oct 2025.

**Result:** gradient boosting on price per mile, **MAE $40.9 (1.74%)** on the August fold and **$42.0 (1.80%)** on the Sep–Oct fold, against $175–193 (≈8%) for a simple baseline.

📄 Full write-up: [`report/Spotter_Report_Mohamed_Amin_Makni.pdf`](report/Spotter_Report_Mohamed_Amin_Makni.pdf) · 🎥 Video walkthrough: [loom.com/share/479f999f835348f6b295c3be99688f9d](https://www.loom.com/share/479f999f835348f6b295c3be99688f9d)

## Repository

| Path | What it is |
|---|---|
| `freight_rate_prediction.ipynb` | The full analysis: data exploration and fixes, validation, models, predictions |
| `validation_predictions.csv` | Final predictions for the 12,000 validation loads (`load_id,predicted_rate`) |
| `outputs/december_chart_inputs.csv` | The 31 December inputs with `predicted_rate` filled |
| `outputs/candidate_december.png` | December chart produced by Spotter's `score.py` |
| `report/` | Report (PDF): validation and split approach, results, December chart |
| `data/` | Put Spotter's data files here (not redistributed, see `data/README.md`) |

## How to run

Python 3.10+ (tested on Python 3.12).

```bash
python -m pip install -r requirements.txt
# copy Spotter's CSV files into data/ (see data/README.md)
jupyter nbconvert --to notebook --execute --inplace freight_rate_prediction.ipynb
```

Or open the notebook and run all cells. It writes `validation_predictions.csv` and fills `data/december_chart_inputs.csv`. Then, with Spotter's `score.py` in the same folder:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

## Approach in short

1. **Understand and fix each feature.** No data dictionary was provided, so each column was explored before cleaning. Distance was checked first (99.8% of loads within ±12% of their lane median), because prices are judged against it.
2. **Data quality.**
   - 677 corrupted prices (1.4%, about 3.5× too low or too high) were removed from training.
   - 292 negative weights are sign errors: same distribution as normal weights, Kolmogorov–Smirnov p = 0.70. Fixed with `abs()`.
   - Missing values: weight → training median, `market_index` → same-day average.
3. **`quote_signal` is a trap.** It correlates ±0.99 with price per mile, but its sign flips between months and it is pure noise in August. A label-free test (does it still separate Reefer from Dry Van?) shows Nov–Dec behave like August, so it was excluded. `market_index` drifts after July and was excluded too.
4. **Time-based validation, never random.**
   - Fold A: train Jan–Jul → test August, the month that behaves like Nov–Dec.
   - Fold B: train Jan–Aug → test Sep–Oct, the most recent data.
   - The final model is retrained on all clean Jan–Oct data.
5. **Model.** scikit-learn `HistGradientBoostingRegressor` predicting the **price per mile**, then multiplied by distance.
   - Features: distance, equipment, weight, pickup and delivery coordinates, day of week.
   - Coordinates let the model handle the 8 validation cities never seen in training: hiding 8 cities from training raises the error only from 1.81% to 1.90%.

| Model | Fold A (Aug) | Fold B (Sep–Oct) |
|---|---|---|
| Baseline: median price per mile by equipment × distance | $193.4 · 8.20% | $175.1 · 8.09% |
| **Final model** | **$40.9 · 1.74%** | **$42.0 · 1.80%** |
| + `market_index` | $66.6 · 2.85% | $116.0 · 4.83% |
| + `quote_signal` | $110.4 · 5.33% | $32.4 · 1.33% |

## Author

Mohamed Amin Makni · [linkedin.com/in/makni-med-amin](https://www.linkedin.com/in/makni-med-amin) · [med-amin-makni.vercel.app](https://med-amin-makni.vercel.app)
