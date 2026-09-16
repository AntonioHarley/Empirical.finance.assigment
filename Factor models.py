from pathlib import Path
import io
import re

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro

folder = Path(__file__).parent

paths = {
    "returns":          folder / "Data" / "49_Industry_Portfolios_Monthly.csv",
    "ff3":              folder / "Data" / "F-F_Research_Data_Factors.csv",
    "ff5":              folder / "Data" / "F-F_Research_Data_5_Factors_2x3.csv",
    "momentum":         folder / "Data" / "F-F_Momentum_Factor.csv"
}

output_dir =        folder / "Factor models"
capm_dir =          output_dir / "CAPM"
ff5_dir =           output_dir / "FF5"

output_dir.mkdir(exist_ok = True)
capm_dir.mkdir(exist_ok = True)
ff5_dir.mkdir(exist_ok = True)
output_dir.mkdir(parents=True, exist_ok=True)

industries = ["Oil", "Banks", "Txtls", "Toys", "Rtail"]


def read_monthly(path):

    lines = path.read_text(encoding="utf-8-sig",errors="replace").splitlines()

    # Monthly observations start with YYYYMM.
    pattern = r"^\s*\d{6}(?=,|\s)"

    # Find the first monthly observation.
    start = next(
        i for i, line in enumerate(lines)
        if re.match(pattern, line)
    )

    end = next(
        (
            i for i in range(start, len(lines))
            if not re.match(pattern, lines[i])
        ),
        len(lines)
    )

    header = lines[start - 1]
    columns = ["Date"] + header.replace(",", " ").split()

    separator = "," if "," in header else r"\s+"

    raw = pd.read_csv(
        io.StringIO("\n".join(lines[start:end])),
        sep=separator,
        names=columns
    )

    raw["Date"] = pd.to_datetime(
        raw["Date"].astype(str),
        format="%Y%m"
    )

    return_columns = raw.columns.drop("Date")
    for column in return_columns:
        raw[column] = pd.to_numeric(raw[column],errors="coerce")
        raw.loc[raw[column].isin([-99.99, -999]),column] = np.nan

    start_date = pd.Timestamp("1963-07-01")

    monthly_data = (
        raw.loc[raw["Date"] >= start_date]
        .set_index("Date")
        .sort_index()
    )

    return monthly_data

monthly_data = {}

for name, path in paths.items():
    data = read_monthly(path)
    monthly_data[name] = data
    data.to_csv(output_dir / f"monthly_{name}.csv")
    print(f"{name}: {len(data)} months loaded")


capm_data = (
    monthly_data["returns"][industries]
    .join(
        monthly_data["ff3"][["Mkt-RF", "RF"]],
        how="inner"
    )
    .join(
        monthly_data["momentum"][["Mom"]],
        how="inner"
    )
    .loc["1963-07":]
    .dropna()
)

capm_corr = capm_data.corr()
capm_data.to_csv(capm_dir / "capm_data.csv")
capm_corr.to_csv(capm_dir / "capm_corr.csv")
print(capm_data.head())
print(capm_corr)

print(
    f"CAPM sample: {capm_data.index.min()} "
    f"to {capm_data.index.max()}"
)
print(f"Observations: {len(capm_data)}")

capm_stats = pd.DataFrame({
    "N":               capm_data.count(),
    "Mean (%)":        capm_data.mean().round(2),
    "Std. dev. (%)":   capm_data.std(ddof=1).round(4),
    "Min (%)":         capm_data.min().round(2),
    "25th pct. (%)":   capm_data.quantile(0.25).round(2),
    "Median (%)":      capm_data.median().round(2),
    "75th pct. (%)":   capm_data.quantile(0.75).round(2),
    "Max (%)":         capm_data.max().round(2),
    "Skewness":        capm_data.skew().round(3),
    "Excess kurtosis": capm_data.kurt().round(3),
    "Shapiro (P-value)": capm_data.apply(lambda x: shapiro(x.dropna()).pvalue).round(4),
})
capm_stats.to_csv(capm_dir / "capm_descriptive_statistics.csv")
print(capm_stats)

excess_returns = capm_data[industries].sub(
    capm_data["RF"], axis=0
)

excess_returns["Mom"] = capm_data["Mom"]


X = sm.add_constant(capm_data[["Mkt-RF"]])
results = []

for portfolio in excess_returns.columns:
    y = excess_returns[portfolio]

    fit = sm.OLS(y, X).fit(use_t=False)

    results.append({
        "Portfolio": portfolio,
        "Alpha (%/month)": fit.params["const"],
        "Alpha p-value": fit.pvalues["const"],
        "Beta": fit.params["Mkt-RF"],
        "R²": fit.rsquared,
        "Adj R²": fit.rsquared_adj,
    })



capm_results = pd.DataFrame(results)
capm_results.to_csv(capm_dir / "capm_regression.csv")


print(capm_results.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}",
    formatters={
        "Alpha p-value": lambda p: (
            "<0.0001" if p < 0.0001 else f"{p:.4f}"
        )
    }
))


ff5_factors = [
    "Mkt-RF",
    "SMB",
    "HML",
    "RMW",
    "CMA"
]



ff5_data = (
    monthly_data["returns"][industries]
    .join(
        monthly_data["ff5"][
            ff5_factors + ["RF"]
        ],
        how="inner"
    )
    .join(
        monthly_data["momentum"][["Mom"]],
        how="inner"
    )
    .loc["1963-07":]
    .dropna()
)



ff5_corr = ff5_data.corr()
ff5_data.to_csv(ff5_dir / "ff5_data.csv")
ff5_corr.to_csv(ff5_dir / "ff5_corr.csv")


ff5_stats = pd.DataFrame({               
    "N":                    ff5_data.count(),
    "Mean (%)":             ff5_data.mean().round(2),
    "Std. dev. (%)":        ff5_data.std(ddof=1).round(4),
    "Min (%)":              ff5_data.min().round(2),
    "25th pct. (%)":        ff5_data.quantile(0.25).round(2),
    "Median (%)":           ff5_data.median().round(2),
    "75th pct. (%)":        ff5_data.quantile(0.75).round(2),
    "Max (%)":              ff5_data.max().round(2),
    "Skewness":             ff5_data.skew().round(3),
    "Excess kurtosis":      ff5_data.kurt().round(3),
    "Shapiro (P-value)":    ff5_data.apply(lambda x: shapiro(x.dropna()).pvalue).round(4)
})


ff5_stats.to_csv(ff5_dir / "ff5_descriptive_statistics.csv")

ff5_excess_returns = (
    ff5_data[industries]
    .sub(
        ff5_data["RF"],
        axis=0
    )
)


ff5_excess_returns["Mom"] = ff5_data["Mom"]
X_ff5 = sm.add_constant(ff5_data[ff5_factors])


ff5_results = []

for portfolio in ff5_excess_returns.columns:
    y = ff5_excess_returns[portfolio]
    fit = sm.OLS(y,X_ff5).fit(use_t=False)

    ff5_results.append({
        "Portfolio":        portfolio,
        "Alpha (%/month)":  fit.params["const"],
        "Alpha p-value":    fit.pvalues["const"],
        "Beta Mkt-RF":      fit.params["Mkt-RF"],
        "Beta SMB":         fit.params["SMB"],
        "Beta HML":         fit.params["HML"],
        "Beta RMW":         fit.params["RMW"],
        "Beta CMA":         fit.params["CMA"],
        "R²":               fit.rsquared,
        "Adj R²":           fit.rsquared_adj,
    })


ff5_results = pd.DataFrame(ff5_results)
ff5_results.to_csv( ff5_dir / "ff5_regression.csv",index=False)

print(
    ff5_results.to_string(
        index=False,
        float_format=lambda value:
            f"{value:.4f}",
        formatters={
            "Alpha p-value":
                lambda p:
                (
                    "<0.0001"
                    if p < 0.0001
                    else f"{p:.4f}"
                )
        }
    )
)

