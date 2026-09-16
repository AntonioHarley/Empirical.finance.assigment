import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
from pathlib import Path
import io
import statsmodels.api as sm
import calendar

data = Path(__file__).parent / "49_Industry_Portfolios_Daily.csv"
outdir = Path(__file__).parent / "Setting the stage"
histogram_dir = outdir / "histograms"
timeseries_dir = outdir / "timeseries"

outdir.mkdir(exist_ok = True)
histogram_dir.mkdir(exist_ok = True)
timeseries_dir.mkdir(exist_ok = True)
lines = data.read_text(encoding="utf-8", errors = "replace").splitlines()


section = next(i for i, line in enumerate(lines) if lines[i].strip() == "Average Value Weighted Returns -- Daily" in line)
header = section + 1 
end = next(i for i in range(header+1, len(lines)) if lines[i].strip() == "")


text = "\n".join(lines[header:end])

raw = pd.read_csv(io.StringIO(text))
raw = raw.rename(columns={raw.columns[0]: "Date"})
raw["Date"] = pd.to_datetime(raw["Date"].astype(str), format="%Y%m%d")

industries = ["Oil", "Banks", "Txtls", "Toys", "Rtail"]

for industry in industries:
    raw[industries] = raw[industries].apply(
        pd.to_numeric,
        errors="coerce"
    )

    raw[industries] = raw[industries].replace(
        [-99.99, -999],
        np.nan
    )    

start_date = pd.Timestamp("1963-07-01")

daily_returns = (
    raw.loc[raw["Date"] >= start_date]
    .dropna()
    .set_index("Date")
    .sort_index()
)

industry_returns = daily_returns[industries]


stats = pd.DataFrame({
    "N":               industry_returns.count(),
    "Mean (%)":        industry_returns.mean().round(2),
    "Std. dev. (%)":   industry_returns.std(ddof=1).round(4),
    "Min (%)":         industry_returns.min().round(2),
    "25th pct. (%)":   industry_returns.quantile(0.25).round(2),
    "Median (%)":      industry_returns.median().round(2),
    "75th pct. (%)":   industry_returns.quantile(0.75).round(2),
    "Max (%)":         industry_returns.max().round(2),
    "Skewness":        industry_returns.skew().round(3),
    "Excess kurtosis": industry_returns.kurt().round(3),
    "Shapiro (P-value)": industry_returns.apply(lambda x: shapiro(x.dropna()).pvalue).round(4),
})

corr = industry_returns.corr()

print(stats)
stats_path = outdir / "Industry descriptive statistics.csv"
corr_path = outdir / "Industry correlation.csv"

stats.to_csv(stats_path)
corr.to_csv(stats_path)

for industry in industries:
    plt.figure(figsize=(8, 5))
    plt.hist(
        industry_returns[industry].dropna(),
        bins=75,
        density=False
    )

    plt.xlabel("Monthly return (%)")
    plt.ylabel("Frequency")
    plt.title(f"{industry} — Average Value Weighted Monthly Returns")
    plt.tight_layout()
    plt.savefig(
        histogram_dir / f"{industry}_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(industry_returns[industry])
    plt.title(f"{industry} — Monthly Returns")
    plt.xlabel("Date")
    plt.ylabel("Monthly return (%)")


    plt.tight_layout()

    plt.savefig(
        timeseries_dir / f"{industry}_timeseries.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


clean_path = outdir / "selected_industry_returns_value_weighted_daily.csv"
daily_returns.to_csv(clean_path)

periods = {
    "Full": (None, None),
    "1963-1977": ("1963-07-01", "1977-12-31"),
    "1978-2002": ("1978-01-01", "2002-12-31"),
    "2003-present": ("2003-01-01", None),
}

results = {
    "monday": [],
    "month": []
}

for period, (start, end) in periods.items():
    period_returns = industry_returns.loc[start:end]
    effects = [
        ("monday", "Monday", period_returns.index.dayofweek == 0)
    ]

    effects += [
        ("month", calendar.month_abbr[m], period_returns.index.month == m)
        for m in range(1, 13)
    ]

    for effect_type, effect_name, dummy in effects:
        X = pd.DataFrame({
            "const": 1,
            "Dummy": dummy.astype(int)
        }, index=period_returns.index)

        for industry in industries:
            model = sm.OLS(period_returns[industry], X).fit(cov_type="HC0")

            results[effect_type].append({
                "Period": period,
                "Industry": industry,
                "Effect": effect_name,
                "Other mean (%)": model.params["const"],
                "Effect mean (%)":
                    model.params["const"] + model.params["Dummy"],
                "Difference (%)": model.params["Dummy"],
                "t-stat": model.tvalues["Dummy"],
                "p-value": model.pvalues["Dummy"]
            })

monday_results = pd.DataFrame(results["monday"]).round(4)
month_results = pd.DataFrame(results["month"]).round(4)

monday_results.to_csv(
    outdir / "monday_effect_subperiods.csv",
    index=False
)

month_results.to_csv(
    outdir / "month_effect_subperiods.csv",
    index=False
)

