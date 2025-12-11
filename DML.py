
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DML trendmodel (tilrettet til DOF-format) med 5x5 km LAEA grid, artsspecifik simulering og PDF-rapport
-----------------------------------------------------------------------------------------------------
- Matcher kolonnerne:
  Dato;Turtidfra;Turtidtil;Loknr;Loknavn;Artnr;Artnavn;Latin;Sortering;Antal;Koen;Adfkode;Adfbeskrivelse;
  Alderkode;Dragtkode;Dragtbeskrivelse;Obserkode;Fornavn;Efternavn;Obser_by;Medobser;Turnoter;Fuglnoter;Metode;
  Obstidfra;Obstidtil;Hemmelig;Kvalitet;Turid;Obsid;DOF_afdeling;lok_laengdegrad;lok_breddegrad;obs_laengdegrad;
  obs_breddegrad;radius;obser_laengdegrad;obser_breddegrad;column0;filename;year;year_1

- Data læses fra Parquet (ingen CSV-parsing)
- Robust håndtering af decimalkomma i koordinater og numeriske felter
- 5x5 km LAEA grid (EPSG:3035)
- DML: Double Machine Learning (plug-in residualisering) + to-komponent abundance (forekomst + log-count|forekomst)
- Propensity score for interårig confounding
- Artsspecifik simulering (null, ±3.3%, ±6.7%/år; konstante og core-edge-varierende) med årlig stokastik (~6.7%)
- Residual-confounding kalibrering (tau_true = beta0 + beta1 * tau_hat) og justering
- Ensemble-bootstrap (B=100) -> 80% CI
- PDF-rapport med plots og nøgletal

Metodisk reference:
Fink et al. (2023) "A Double machine learning trend model for citizen science data", MEE 14:2435–2448.
"""

import argparse
import os
import glob
import re
import warnings
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyproj import Transformer

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.platypus import Table, TableStyle

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="DML trend (DOF-format) med 5x5 km LAEA grid, artsspecifik simulering og PDF-rapport.")
    ap.add_argument("--data-dir", type=str, required=False, help="Rodmappe med parquet-filer.")
    ap.add_argument("--pattern", type=str, default="observationer_year=*.parquet", help="Glob pattern under data-dir.")
    ap.add_argument("--workers", type=int, default=4, help="Antal parallelle workers til indlæsning.")
    ap.add_argument("--species-field", type=str, default="Artnavn", help="Kolonnenavn til artsfilter (fx 'Latin', 'Artnr', 'Artnavn').")
    ap.add_argument("--species", type=str, default=None, help="Artsnavn/ID til filtrering (valgfri).")
    ap.add_argument("--start-year", type=int, default=None, help="Startår (valgfri).")
    ap.add_argument("--end-year", type=int, default=None, help="Slutår (valgfri).")
    ap.add_argument("--grid-crs", type=str, default="EPSG:3035", help="LAEA proj (standard: EPSG:3035).")
    ap.add_argument("--grid-size-km", type=float, default=5.0, help="Celle-størrelse i km (standard 5).")
    ap.add_argument("--heterogeneous", action="store_true", help="Estimer heterogen trend tau(W) og lav kort.")
    ap.add_argument("--output", type=str, default="trend_resultater.csv", help="CSV med resultater.")
    ap.add_argument("--report-pdf", type=str, default="trend_rapport.pdf", help="PDF-rapport.")
    ap.add_argument("--mode", type=str, choices=["normal", "save_preprocessed", "use_preprocessed"], default="normal",
                    help="Kørselstilstand: normal, save_preprocessed, use_preprocessed.")
    ap.add_argument("--preprocessed-path", type=str, default="output/processed_data.parquet",
                    help="Sti til preprocessed data (bruges af save/use).")
    return ap.parse_args()


# ---------------------------
# Parquet I/O
# ---------------------------

def _read_single_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def read_parquet_parallel(data_dir: str, pattern: str, workers: int, start_year: Optional[int]=None, end_year: Optional[int]=None) -> pd.DataFrame:
    paths = glob.glob(os.path.join(data_dir, pattern))
    # Filtrer på år hvis muligt
    if start_year is not None or end_year is not None:
        filtered_paths = []
        for p in paths:
            m = re.search(r"year=(\d{4})", p)
            if m:
                y = int(m.group(1))
                if (start_year is None or y >= start_year) and (end_year is None or y <= end_year):
                    filtered_paths.append(p)
        paths = filtered_paths
    if len(paths) == 0:
        raise FileNotFoundError(f"Ingen filer matcher: {os.path.join(data_dir, pattern)}")
    print(f"Læser {len(paths)} parquet-filer med {workers} workers...")
    dfs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_read_single_parquet, p): p for p in paths}
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            dfs.append(result)
            print(f"  [{i+1}/{len(paths)}] {futures[fut]}")
    return pd.concat(dfs, ignore_index=True)


# ---------------------------
# Hjælpere til formater (decimalkomma, tid)
# ---------------------------

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # Dato som 'YYYY-MM-DD' (eksempel), men accepter også andre varianter; dag-først tillades
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def _time_to_minutes(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return s
    m = re.match(r"^\s*(\d{1,2}):(\d{2})", str(s))
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return np.nan

def _count_observers(s):
    if pd.isna(s) or str(s).strip() == "":
        return 1
    return max(1, len(re.split(r"[;,]", str(s).strip())))

def _to_num_comma(x):
    """Konverter str med decimalkomma til float. Bevarer floats/int som de er."""
    if isinstance(x, (int, float, np.number)):
        return float(x)
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    # Fjern tusindtalsprik hvis den findes (sjældent her), og erstat komma med punktum
    s = s.replace(".", "").replace(",", ".") if ("," in s and "." in s and s.find(",") > s.find(".")) else s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


# ---------------------------
# Feature engineering (tilpasset kolonner)
# ---------------------------

REQ_COLS = [
    "Dato","Turtidfra","Turtidtil","Loknr","Loknavn","Artnr","Artnavn","Latin","Sortering","Antal","Koen",
    "Adfkode","Adfbeskrivelse","Alderkode","Dragtkode","Dragtbeskrivelse","Obserkode","Fornavn","Efternavn",
    "Obser_by","Medobser","Turnoter","Fuglnoter","Metode","Obstidfra","Obstidtil","Hemmelig","Kvalitet",
    "Turid","Obsid","DOF_afdeling","lok_laengdegrad","lok_breddegrad","obs_laengdegrad","obs_breddegrad",
    "radius","obser_laengdegrad","obser_breddegrad","column0","filename","year","year_1"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sikr kolonner findes (tillad manglende obs_* / obser_* koordinater)
    missing = [c for c in REQ_COLS if (c not in df.columns)]
    if missing:
        raise ValueError(f"Mangler forventede kolonner: {missing}")

    df = df.copy()

    # Dato -> år, DOY, ugedag, weekend
    df["Dato_dt"] = _safe_to_datetime(df["Dato"])
    df["year_from_date"] = df["Dato_dt"].dt.year
    # Fallback til 'year' hvis Dato ikke kunne parses
    df["year"] = df["year_from_date"].where(df["year_from_date"].notna(), pd.to_numeric(df["year"], errors="coerce"))
    df["doy"] = df["Dato_dt"].dt.dayofyear
    df["weekday"] = df["Dato_dt"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype("Int64").fillna(0).astype(int)

    # Observationstid
    df["obs_min_from"] = df["Obstidfra"].map(_time_to_minutes)
    df["obs_min_to"] = df["Obstidtil"].map(_time_to_minutes)
    df["obs_duration_min"] = (df["obs_min_to"] - df["obs_min_from"])
    df["obs_duration_min"] = df["obs_duration_min"].fillna(0).clip(lower=0)

    # Turtid
    df["tour_min_from"] = df["Turtidfra"].map(_time_to_minutes)
    df["tour_min_to"] = df["Turtidtil"].map(_time_to_minutes)
    df["tour_duration_min"] = (df["tour_min_to"] - df["tour_min_from"])
    df["tour_duration_min"] = df["tour_duration_min"].fillna(0).clip(lower=0)

    df["n_observers"] = df["Medobser"].map(_count_observers).astype(float)
    df["time_of_day_frac"] = df["obs_min_from"] / (24*60)

    # Numerik med decimalkomma
    num_cols_comma = [
        "lok_laengdegrad","lok_breddegrad","obs_laengdegrad","obs_breddegrad",
        "obser_laengdegrad","obser_breddegrad","Antal","radius"
    ]
    for col in num_cols_comma:
        if col in df.columns:
            df[col] = df[col].map(_to_num_comma)

    # Primære koordinater: brug obs_* hvis tilgængelig, ellers lok_*
    df["lat"] = df["obs_breddegrad"].where(df["obs_breddegrad"].notna(), df["lok_breddegrad"])
    df["lon"] = df["obs_laengdegrad"].where(df["obs_laengdegrad"].notna(), df["lok_laengdegrad"])

    # Response
    df["Antal"] = pd.to_numeric(df["Antal"], errors="coerce").fillna(0).clip(lower=0)
    df["detected"] = (df["Antal"] > 0).astype(int)
    df["log_count"] = np.log1p(df["Antal"])

    return df

def engineer_features_parallel(df: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    chunk_size = int(np.ceil(len(df) / workers))
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]
    print(f"Feature-engineering i {workers} parallelle chunks...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(engineer_features, chunk) for chunk in chunks]
        dfs = []
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            dfs.append(fut.result())
            print(f"  Chunk {i+1}/{workers} færdig")
    return pd.concat(dfs, ignore_index=True)

def filter_by_species_year(df: pd.DataFrame,
                           species_field: str,
                           species_value: Optional[str],
                           start_year: Optional[int],
                           end_year: Optional[int]) -> pd.DataFrame:
    out = df.copy()
    if species_value is not None:
        if species_field not in out.columns:
            raise ValueError(f"species_field '{species_field}' findes ikke i data.")
        out = out[out[species_field] == species_value]
    if start_year is not None:
        out = out[out["year"] >= start_year]
    if end_year is not None:
        out = out[out["year"] <= end_year]
    out = out.dropna(subset=["year", "lat", "lon"])
    return out


# ---------------------------
# LAEA 5x5 km grid
# ---------------------------

def project_to_laea(df: pd.DataFrame, crs: str) -> pd.DataFrame:
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(df["lon"].values, df["lat"].values)
    df = df.copy()
    df["x_laea"] = x
    df["y_laea"] = y
    return df

def assign_grid_cells(df: pd.DataFrame, cell_km: float) -> pd.DataFrame:
    cell_m = cell_km * 1000.0
    df = df.copy()
    df["ix"] = np.floor_divide(df["x_laea"], cell_m).astype(int)
    df["iy"] = np.floor_divide(df["y_laea"], cell_m).astype(int)
    df["grid_id"] = df["ix"].astype(str) + "_" + df["iy"].astype(str)
    df["x_c"] = (df["ix"] + 0.5) * cell_m
    df["y_c"] = (df["iy"] + 0.5) * cell_m
    return df


# ---------------------------
# DML kerne
# ---------------------------

def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols_effort = ["obs_duration_min", "tour_duration_min", "n_observers", "time_of_day_frac"]
    feature_cols_temporal = ["doy", "weekday", "is_weekend"]
    feature_cols_spatial = ["x_c", "y_c"]  # LAEA
    X_cols = feature_cols_effort + feature_cols_temporal + feature_cols_spatial
    W_cols = feature_cols_spatial
    X = df[X_cols].copy()
    W = df[W_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    W = W.fillna(W.median(numeric_only=True))
    return X, W

def fit_propensity_score(X: pd.DataFrame, T: np.ndarray) -> np.ndarray:
    model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300)
    model.fit(X, T)
    return model.predict(X)

def fit_conditional_mean_reg(X: pd.DataFrame, Y: np.ndarray) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=42)
    model.fit(X, Y)
    return model.predict(X)

def dml_global_tau(Y: np.ndarray, T: np.ndarray, mX: np.ndarray, sX: np.ndarray) -> Tuple[float, float]:
    y_res = Y - mX
    t_res = T - sX
    lr = LinearRegression(fit_intercept=False)
    lr.fit(t_res.reshape(-1, 1), y_res)
    tau = float(lr.coef_[0])
    r2 = lr.score(t_res.reshape(-1, 1), y_res)
    return tau, r2

def rlearner_tau_w(Y: np.ndarray, T: np.ndarray, mX: np.ndarray, sX: np.ndarray, W: pd.DataFrame) -> np.ndarray:
    Z = (T - sX)
    ytilde = (Y - mX)
    eps = np.percentile(np.abs(Z), 5)
    keep = np.abs(Z) > eps
    if keep.sum() < 200:
        tau_global, _ = dml_global_tau(Y, T, mX, sX)
        return np.full(len(Y), tau_global)
    R = ytilde[keep] / Z[keep]
    W_keep = W.iloc[keep]
    rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=42)
    rf.fit(W_keep, R, sample_weight=(Z[keep] ** 2))
    tau_w = rf.predict(W)
    return tau_w


def _year_design(df: pd.DataFrame, X: pd.DataFrame):
    """Byg en one-hot designmatrix for år og residualiser årsdummies mod X."""
    years = np.sort(df["year"].astype(int).unique())
    y0 = int(years.min())
    year_labels = df["year"].astype(int).values
    # One-hot D (n_obs x n_years)
    Ymap = {y:i for i,y in enumerate(years)}
    D = np.zeros((len(df), len(years)), dtype=float)
    for i, y in enumerate(year_labels):
        D[i, Ymap[y]] = 1.0
    # P(year|X) via histGB classifier
    clf_year = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300)
    clf_year.fit(X, year_labels)
    # Align predict_proba columns to 'years' order
    proba = clf_year.predict_proba(X)
    # sklearn kan returnere klasser i sorteret rækkefølge -> match:
    cls_order = clf_year.classes_
    P = np.zeros_like(D)
    for j, y in enumerate(years):
        cj = np.where(cls_order == y)[0][0]
        P[:, j] = proba[:, cj]
    D_res = D - P
    return years, y0, D_res

def _ridge_year_effects(y_res: np.ndarray, D_res: np.ndarray, lambda_diff: float = 5.0):
    """Ridge-penaliseret estimation af års-koefficienter med glathed via første-difference."""
    n_years = D_res.shape[1]
    # Penalty: ||Δβ||^2, hvor Δ er (n_years-1) x n_years difference-matrix
    Delta = np.zeros((n_years-1, n_years))
    for r in range(n_years-1):
        Delta[r, r] = -1.0
        Delta[r, r+1] = 1.0
    # Normal equations: (D' D + λ Δ'Δ) β = D' y
    XtX = D_res.T @ D_res
    Pen = Delta.T @ Delta
    A = XtX + lambda_diff * Pen
    b = D_res.T @ y_res
    # Identifiabilitet: sæt basisåret (kolonne 0) til 0 via "ref" - fjern kol., løs og indsæt 0
    A_sub = A[1:, 1:]
    b_sub = b[1:]
    beta_sub = np.linalg.solve(A_sub, b_sub)
    beta = np.zeros(n_years)
    beta[1:] = beta_sub
    return beta  # log-skala offsets pr. år (baseline år = 0)


# ---------------------------
# To-komponent abundance (forekomst + log-count|forekomst)
# ---------------------------

def estimate_trend_components(df: pd.DataFrame, heterogeneous: bool = False, ensemble_B: int = 100) -> Dict[str, float]:
    # Forekomst
    X_det, W_det = build_feature_matrix(df)
    Y_det = df["detected"].values.astype(float)
    T_det = df["year"].values.astype(float)

    sX_det = fit_propensity_score(X_det, T_det)
    clf = RandomForestClassifier(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=42)
    clf.fit(X_det, df["detected"].values.astype(int))
    proba = clf.predict_proba(X_det)
    if proba.shape[1] == 1:
        mX_det = np.full(len(X_det), proba[0, 0])
    else:
        mX_det = proba[:, 1]

    tau_det_global, r2_det = dml_global_tau(Y_det, T_det, mX_det, sX_det)
    tau_det_w = rlearner_tau_w(Y_det, T_det, mX_det, sX_det, W_det) if heterogeneous else None

    # Count|forekomst
    df_pos = df[df["detected"] == 1]
    if len(df_pos) < 100:
        tau_cnt_global, r2_cnt, tau_cnt_w = 0.0, np.nan, None
    else:
        X_cnt, W_cnt = build_feature_matrix(df_pos)
        Y_cnt = df_pos["log_count"].values.astype(float)
        T_cnt = df_pos["year"].values.astype(float)
        sX_cnt = fit_propensity_score(X_cnt, T_cnt)
        mX_cnt = fit_conditional_mean_reg(X_cnt, Y_cnt)
        tau_cnt_global, r2_cnt = dml_global_tau(Y_cnt, T_cnt, mX_cnt, sX_cnt)
        tau_cnt_w = rlearner_tau_w(Y_cnt, T_cnt, mX_cnt, sX_cnt, W_cnt) if heterogeneous else None

    tau_abund_log = tau_det_global + tau_cnt_global
    ppy_abund = 100.0 * (np.exp(tau_abund_log) - 1.0)
    ppy_det = 100.0 * (np.exp(tau_det_global) - 1.0)
    ppy_cnt = 100.0 * (np.exp(tau_cnt_global) - 1.0)


    # Ensemble-bootstrap (80% CI)
    rng = np.random.default_rng(42)
    ppy_samples = []
    tau_samples = []   # <- NEW: saml log-trends fra hver resample
    n = len(df)
    for b in range(ensemble_B):
        if (b+1) % 10 == 0 or b == 0 or b == ensemble_B-1:
            print(f"  Bootstrap {b+1}/{ensemble_B} ...")
        idx = rng.integers(low=0, high=n, size=n)
        df_b = df.iloc[idx]

        # det
        Xd, _ = build_feature_matrix(df_b)
        Td = df_b["year"].values.astype(float)
        sXd = fit_propensity_score(Xd, Td)
        clf_b = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, n_jobs=-1)
        clf_b.fit(Xd, df_b["detected"].values.astype(int))
        proba = clf_b.predict_proba(Xd)
        # Robust fallback hvis kun én klasse
        if proba.shape[1] == 1:
            mXd = np.full(len(Xd), proba[0, 0])
        else:
            mXd = proba[:, 1]
        tau_d, _ = dml_global_tau(df_b["detected"].values.astype(float), Td, mXd, sXd)

        # cnt
        df_b_pos = df_b[df_b["detected"] == 1]
        if len(df_b_pos) < 50:
            tau_c = 0.0
        else:
            Xc, _ = build_feature_matrix(df_b_pos)
            Tc = df_b_pos["year"].values.astype(float)
            sXc = fit_propensity_score(Xc, Tc)
            mXc = fit_conditional_mean_reg(Xc, df_b_pos["log_count"].values.astype(float))
            tau_c, _ = dml_global_tau(df_b_pos["log_count"].values.astype(float), Tc, mXc, sXc)

        tau_abund_b = tau_d + tau_c             # <- NEW
        tau_samples.append(tau_abund_b)         # <- NEW
        ppy_samples.append(100.0 * (np.exp(tau_abund_b) - 1.0))

    lo80, hi80 = np.percentile(ppy_samples, [10, 90])

    out = {
        "tau_det_global": float(tau_det_global),
        "tau_cnt_global": float(tau_cnt_global),
        "tau_abund_global_log": float(tau_abund_log),
        "PPY_abundance_global": float(ppy_abund),
        "PPY_abundance_CI80_low": float(lo80),
        "PPY_abundance_CI80_high": float(hi80),
        "PPY_detection_global": float(ppy_det),
        "PPY_count_global": float(ppy_cnt),
        "tau_samples": np.array(tau_samples),   # <- NEW: returnér alle bootstrap log-trends
    }


    if heterogeneous:
        print("Beregner rumlig trend-tabel (tau_grid_table)...")
        df_tmp = df[["grid_id", "x_c", "y_c"]].copy()
        if tau_det_w is not None:
            df_tmp["tau_det_w"] = tau_det_w
        if (df_pos is not None) and (len(df_pos) > 0) and ("tau_cnt_w" in locals()) and (tau_cnt_w is not None):
            df_cnt_tmp = df_pos[["grid_id"]].copy()
            df_cnt_tmp["tau_cnt_w"] = tau_cnt_w
            g_cnt = df_cnt_tmp.groupby("grid_id", as_index=False)["tau_cnt_w"].mean()
        else:
            g_cnt = pd.DataFrame(columns=["grid_id", "tau_cnt_w"])

        g_det = df_tmp.groupby("grid_id", as_index=False).agg(
            x_c=("x_c", "median"), y_c=("y_c", "median"),
            tau_det_w=("tau_det_w", "mean")
        )
        g = g_det.merge(g_cnt, on="grid_id", how="left")
        g["tau_abund_w"] = g["tau_det_w"].fillna(0.0) + g["tau_cnt_w"].fillna(0.0)
        out["tau_grid_table"] = g
        print(f"Rumlig trend-tabel færdig: {len(g)} grids")
    return out


def estimate_yearwise_components(df: pd.DataFrame, lambda_diff: float = 5.0):
    """
    Estimer års-specifikke effekter (ikke-monotont indeks) for både forekomst og count|forekomst.
    Returnerer DataFrame med year, beta_det, beta_cnt, beta_abund, index_raw (baseline=100).
    """
    # Fælles feature-matrix
    X_det, _ = build_feature_matrix(df)
    # Forekomst
    Y_det = df["detected"].values.astype(float)
    # Residualiser outcome (mX) og årsdummies (D_res)
    clf = RandomForestClassifier(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=42)
    clf.fit(X_det, df["detected"].values.astype(int))
    mX_det = clf.predict_proba(X_det)[:, 1]
    y_res_det = Y_det - mX_det
    years, y0, D_res_det = _year_design(df, X_det)

    beta_det = _ridge_year_effects(y_res_det, D_res_det, lambda_diff=lambda_diff)

    # Count|forekomst
    df_pos = df[df["detected"] == 1]
    if len(df_pos) < 100:
        beta_cnt = np.zeros_like(beta_det)
    else:
        X_cnt, _ = build_feature_matrix(df_pos)
        Y_cnt = df_pos["log_count"].values.astype(float)
        mX_cnt = fit_conditional_mean_reg(X_cnt, Y_cnt)  # RF
        y_res_cnt = Y_cnt - mX_cnt
        # Align D_res til df_pos (samme år-rækkefølge)
        years_cnt = df_pos["year"].astype(int).values
        Ymap = {y:i for i,y in enumerate(years)}
        D_res_cnt = np.zeros((len(df_pos), len(years)), dtype=float)
        # recompute P(year|X) for df_pos via år-klf på X_cnt:
        clf_year_cnt = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300)
        clf_year_cnt.fit(X_cnt, df_pos["year"].astype(int).values)
        proba_cnt = clf_year_cnt.predict_proba(X_cnt)
        cls_order_cnt = clf_year_cnt.classes_
        for i, y in enumerate(years_cnt):
            j = Ymap[y]
            # hent rette kolonne for år y i proba_cnt:
            cj = np.where(cls_order_cnt == y)[0][0]
            D_res_cnt[i, j] = 1.0 - proba_cnt[i, cj]
            # (for andre år, residualet er -p_j; det svarer til de 0'er i one-hot minus p_j)
            for k, yk in enumerate(years):
                if yk != y:
                    ck = np.where(cls_order_cnt == yk)[0]
                    if len(ck) == 1:
                        D_res_cnt[i, k] = -proba_cnt[i, ck[0]]
                    else:
                        D_res_cnt[i, k] += 0.0  # hvis yk ikke forekommer i klasserne
        beta_cnt = _ridge_year_effects(y_res_cnt, D_res_cnt, lambda_diff=lambda_diff)

    beta_abund = beta_det + beta_cnt  # log-skala
    index = 100.0 * np.exp(beta_abund)  # baseline år (første i 'years') = 100

    out = pd.DataFrame({
        "year": years,
        "beta_det": beta_det,
        "beta_cnt": beta_cnt,
        "beta_abund": beta_abund,
        "index_raw": index
    })
    return out, y0


# ---------------------------
# Artsspecifik simulering + RC-kalibrering
# ---------------------------

def compute_grid_baseline(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("grid_id").agg(
        x_c=("x_c", "median"),
        y_c=("y_c", "median"),
        det_rate=("detected", "mean"),
        mean_logcount_pos=("log_count", lambda s: s[s > 0].mean() if (s > 0).any() else 0.0),
        n=("grid_id", "size")
    ).reset_index()
    # skaler til [0,1]
    dr = (g["det_rate"] - g["det_rate"].min()) / (g["det_rate"].max() - g["det_rate"].min() + 1e-9)
    lc = (g["mean_logcount_pos"] - g["mean_logcount_pos"].min()) / (g["mean_logcount_pos"].max() - g["mean_logcount_pos"].min() + 1e-9)
    g["core_index"] = 0.6*dr + 0.4*lc
    return g

def assign_spatial_trend(ggrid: pd.DataFrame, scenario: str, magnitude_ppy: float) -> pd.Series:
    log_rate = np.log1p(magnitude_ppy/100.0)
    core = ggrid["core_index"].fillna(0.0)
    if scenario == "constant_pos":
        tau_g = np.full(len(ggrid), log_rate)
    elif scenario == "constant_neg":
        tau_g = np.full(len(ggrid), -log_rate)
    elif scenario == "vary_core_pos":
        tau_g = (0.3*log_rate) + (0.7*log_rate * core.values)
    elif scenario == "vary_core_neg":
        tau_g = -(0.3*log_rate + 0.7*log_rate * core.values)
    else:
        tau_g = np.zeros(len(ggrid))
    return pd.Series(tau_g, index=ggrid["grid_id"])

def _logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def _expit(x):
    x = np.clip(x, -12, 12)
    return 1.0/(1.0 + np.exp(-x))

def simulate_species_datasets(df: pd.DataFrame, reps_per_scenario: int = 5) -> List[Dict]:
    rng = np.random.default_rng(123)
    print("Starter artsspecifik simulering...")
    # Baseline m(X) modeller
    print("  Træner baseline RandomForest for forekomst...")
    X, _ = build_feature_matrix(df)
    clf = RandomForestClassifier(n_estimators=600, min_samples_leaf=5, n_jobs=-1, random_state=11)
    clf.fit(X, df["detected"].values.astype(int))

    # Core/edge-profil pr grid
    print("  Beregner core/edge-profil pr grid...")
    ggrid = compute_grid_baseline(df)

    years = np.sort(df["year"].unique())
    t0 = years.min()
    sigma = 0.067  # ~6.7%

    scenarios = [
        ("null", 0.0, "null"),
        ("const_pos_3.3", 3.3, "constant_pos"),
        ("const_neg_3.3", 3.3, "constant_neg"),
        ("const_pos_6.7", 6.7, "constant_pos"),
        ("const_neg_6.7", 6.7, "constant_neg"),
        ("vary_core_pos_3.3", 3.3, "vary_core_pos"),
        ("vary_core_neg_3.3", 3.3, "vary_core_neg"),
        ("vary_core_pos_6.7", 6.7, "vary_core_pos"),
        ("vary_core_neg_6.7", 6.7, "vary_core_neg"),
    ]

    results = []
    for i, (name, mag, sc) in enumerate(scenarios):
        print(f"Scenario {i+1}/{len(scenarios)}: {name} (mag={mag}, type={sc})")
        tau_g = assign_spatial_trend(ggrid, sc, mag)
        for r in range(reps_per_scenario):
            print(f"  Rep {r+1}/{reps_per_scenario} for scenario '{name}'")
            # Stratificeret resampling pr år
            parts = []
            for yr in years:
                block = df[df["year"] == yr]
                idx = rng.integers(low=0, high=len(block), size=len(block))
                parts.append(block.iloc[idx])
            sim = pd.concat(parts, ignore_index=True)

            X_sim, _ = build_feature_matrix(sim)
            proba = clf.predict_proba(X_sim)
            if proba.shape[1] == 1:
                p_det_base = np.full(len(X_sim), proba[0, 0])
            else:
                p_det_base = proba[:, 1]

            tau_obs = sim["grid_id"].map(tau_g).fillna(0.0).values
            tt = (sim["year"].values - t0).astype(float)

            # Stokastik pr (grid,year)
            gy = sim[["grid_id", "year"]].astype(str).agg("|".join, axis=1)
            keys = gy.unique()
            eps_map = {k: rng.normal(0.0, sigma) for k in keys}
            eps = np.array([eps_map[k] for k in gy])

            # Forekomst: logit(p*) = logit(p0) + tau*tt + eps
            p_det_star = _expit(_logit(p_det_base) + tau_obs*tt + eps)
            detected_star = (rng.random(len(p_det_star)) < p_det_star).astype(int)

            # Count baseline på sim-data
            mX_log_sim = fit_conditional_mean_reg(X_sim, sim["log_count"].values.astype(float))
            log_count_star_full = np.maximum(mX_log_sim + tau_obs*tt + eps, 0.0)

            # DML på syntetiske data
            # Forekomst
            sX_det = fit_propensity_score(X_sim, sim["year"].values.astype(float))
            mX_det = p_det_base
            tau_det_hat, _ = dml_global_tau(detected_star.astype(float), sim["year"].values.astype(float), mX_det, sX_det)

            # Count|forekomst
            sel = detected_star == 1
            if sel.sum() < 50:
                tau_cnt_hat = 0.0
            else:
                X_cnt = X_sim[sel]
                sX_cnt = fit_propensity_score(X_cnt, sim["year"].values.astype(float)[sel])
                y_cnt_star = log_count_star_full[sel]
                mX_cnt = fit_conditional_mean_reg(X_cnt, y_cnt_star)
                tau_cnt_hat, _ = dml_global_tau(y_cnt_star, sim["year"].values.astype(float)[sel], mX_cnt, sX_cnt)

            tau_abund_hat = tau_det_hat + tau_cnt_hat  # log-skala

            # "Sand" global trend ≈ vægtet middel af tau_g efter grid-frekvens
            w = sim.groupby("grid_id").size()
            w = w / w.sum()
            true_global = float((w * tau_g.loc[w.index]).sum())

            print(f"  Rep {r+1} færdig: tau_hat={tau_abund_hat:.4f}, tau_true={true_global:.4f}")
            results.append({"scenario": name, "tau_true": true_global, "tau_hat": float(tau_abund_hat)})
    print("Artsspecifik simulering færdig.")
    return results

def residual_confounding_via_simulation(df: pd.DataFrame, reps_per_scenario: int = 5) -> Tuple[float, float, pd.DataFrame]:
    sims = simulate_species_datasets(df, reps_per_scenario=reps_per_scenario)
    sim_df = pd.DataFrame(sims)
    lr = LinearRegression()
    lr.fit(sim_df[["tau_hat"]], sim_df["tau_true"])
    beta0 = float(lr.intercept_); beta1 = float(lr.coef_[0])
    return beta0, beta1, sim_df


def bootstrap_yearwise(df: pd.DataFrame,
                       beta0: float,
                       beta1: float,
                       B: int = 100,
                       lambda_diff: float = 5.0,
                       random_state: int = 123):
    """
    Stratificeret bootstrap pr. år for at få 80% CI på slope-RC årskurven.
    - Resampler med tilbagelægning inden for hvert år (samme antal som originalen).
    - Genestimerer årseffekter via estimate_yearwise_components(...).
    - RC-kalibrerer årlige slopes (Δβ_y), sætter første år til 0, akkumulerer og
      konverterer til indeks (baseline=100).
    Returnerer: years, lo80, hi80 (arrays i samme orden).
    """
    rng = np.random.default_rng(random_state)
    years = np.sort(df["year"].astype(int).unique())
    n_years = len(years)
    year_to_rows = {y: np.where(df["year"].astype(int).values == y)[0] for y in years}

    # Saml bootstrap-indeks pr. replikat (B x n_years)
    idx_mat = np.zeros((B, n_years), dtype=float)

    for b in range(B):
        parts = []
        for y in years:
            rows = year_to_rows[y]
            # robusthed: hvis et år har meget få observationer, resamplér stadig med samme størrelse
            sel = rng.integers(low=0, high=len(rows), size=len(rows))
            parts.append(df.iloc[rows[sel]])
        df_b = pd.concat(parts, ignore_index=True)

        # Årseffekter på bootstrap-sample
        ydf_b, _ = estimate_yearwise_components(df_b, lambda_diff=lambda_diff)
        ydf_b = ydf_b.sort_values("year").reset_index(drop=True)

        # RC på slopes (Δβ_y)
        tau_b = ydf_b["beta_abund"].diff().fillna(0.0).values          # Δβ_y
        tau_b_rc = beta0 + beta1 * tau_b
        if len(tau_b_rc) > 0:
            tau_b_rc[0] = 0.0                                          # baseline år = 0
        beta_b = np.cumsum(tau_b_rc)
        idx_b = 100.0 * np.exp(beta_b)                                 # baseline=100

        # ydf_b["year"] bør matche 'years' pga. stratificeret resampling
        # ellers kunne vi mappe – men her er det samme orden:
        idx_mat[b, :] = idx_b
        print(f"  Bootstrap {b+1}/{B} færdig")

    lo80 = np.percentile(idx_mat, 10, axis=0)
    hi80 = np.percentile(idx_mat, 90, axis=0)
    return years, lo80, hi80


# ---------------------------
# Plots & PDF-rapport
# ---------------------------

def plot_time_series(df: pd.DataFrame, out_png: str):
    ts = df.groupby("year")["Antal"].mean()
    plt.figure(figsize=(6,3))
    plt.plot(ts.index, ts.values, marker="o")
    plt.title("Gennemsnitligt rapporteret antal pr. år")
    plt.xlabel("År"); plt.ylabel("Antal (gennemsnit)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_components_bar(ppy_det: float, ppy_cnt: float, out_png: str):
    plt.figure(figsize=(4,3))
    vals = [ppy_det, ppy_cnt, ppy_det+ppy_cnt]
    labels = ["Forekomst", "Count|forekomst", "Sum"]
    colors_ = ["#0072B2", "#E69F00", "#009E73"]
    plt.bar(labels, vals, color=colors_)
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("% pr. år (PPY)")
    plt.title("Komponentbidrag til abundance-trend")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_tau_calibration(sim_df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(4.8,4.2))
    plt.scatter(sim_df["tau_hat"], sim_df["tau_true"], s=12, alpha=0.6)
    lims = [min(sim_df["tau_hat"].min(), sim_df["tau_true"].min()),
            max(sim_df["tau_hat"].max(), sim_df["tau_true"].max())]
    plt.plot(lims, lims, 'k--', lw=1, label="1:1")
    plt.xlabel(r"$\hat{\tau}$ (log/år)")
    plt.ylabel(r"$\tau_{\mathrm{true}}$ (log/år)")
    plt.title("Residual confounding kalibrering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def plot_trend_map(grid_table: pd.DataFrame, out_png: str):
    if grid_table is None or len(grid_table)==0:
        return
    plt.figure(figsize=(5.2,5))
    x = grid_table["x_c"].values / 1000.0
    y = grid_table["y_c"].values / 1000.0
    z = grid_table["tau_abund_w"].values
    ppy = 100.0 * (np.exp(z) - 1.0)
    vmax = np.nanpercentile(np.abs(ppy), 95) if np.isfinite(ppy).any() else 5
    sc = plt.scatter(x, y, c=ppy, cmap="coolwarm", s=10, vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(sc)
    cbar.set_label("% pr. år (PPY)")
    plt.title("Rumlig heterogenitet i trend (LAEA km)")
    plt.xlabel("LAEA x (km)"); plt.ylabel("LAEA y (km)")

    # --- Fast aksegrænse for Danmark + 10 km buffer ---
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    # Yderpunkter (lon, lat)
    points = [
        (8.0833, 57.75),   # Nord (Skagen Nordstrand)
        (12.1167, 54.5667),# Syd (Gedser Odde)
        (15.2, 55.3167),   # Øst (Christiansø)
        (8.0833, 55.55),   # Vest (Blåvandshuk, lidt sydligere for bredde)
    ]
    xs, ys = zip(*[transformer.transform(lon, lat) for lon, lat in points])
    x_min, x_max = min(xs)-10000, max(xs)+10000
    y_min, y_max = min(ys)-10000, max(ys)+10000
    plt.xlim(x_min/1000, x_max/1000)
    plt.ylim(y_min/1000, y_max/1000)
    # -------------------------------------------------

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def make_pdf_report(pdf_path: str, meta: Dict[str, str], res: Dict[str, float],
                    sim_df: pd.DataFrame, fig_paths: Dict[str, str]):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    W, H = A4
    margin = 1.5*cm


    # Side 1: Titel og nøglestatistikker
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, H - margin, "DML-trendrapport (5x5 km LAEA)")

    c.setFont("Helvetica", 10)
    y = H - margin - 1.2*cm
    lines = [
        f"Art: {meta.get('species', '')}  |  Felt: {meta.get('species_field','')}",
        f"År: {meta.get('year_min','')}–{meta.get('year_max','')}  |  N={meta.get('n','')}",
        f"Grid: 5×5 km (EPSG:3035, LAEA Europe)",
    ]
    for line in lines:
        c.drawString(margin, y, line); y -= 0.45*cm

    data = [
        ["Metrik", "Værdi"],
        ["Abundance trend (PPY, global)", f"{res['PPY_abundance_global']:.2f} %/år"],
        ["CI80 (lav–høj)", f"{res['PPY_abundance_CI80_low']:.2f} – {res['PPY_abundance_CI80_high']:.2f}"],
        ["RC-justeret (PPY)", f"{meta.get('PPY_RC', np.nan):.2f} %/år"],
        ["Komponent: forekomst (PPY)", f"{res['PPY_detection_global']:.2f} %/år"],
        ["Komponent: count|forekomst (PPY)", f"{res['PPY_count_global']:.2f} %/år"],
        ["RC-koef beta0", f"{meta.get('beta0', np.nan):.4f}"],
        ["RC-koef beta1", f"{meta.get('beta1', np.nan):.4f}"],
    ]
    table = Table(data, colWidths=[8*cm, 6*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ]))
    table.wrapOn(c, W - 2*margin, H)
    table.drawOn(c, margin, y - 5*cm)
    y2 = y - 5.8*cm

    if os.path.exists(fig_paths["ts"]):
        c.drawImage(fig_paths["ts"], margin, y2-7.5*cm, width=8.5*cm, height=7.5*cm, preserveAspectRatio=True)
    if os.path.exists(fig_paths["comp"]):
        c.drawImage(fig_paths["comp"], margin+9*cm, y2-7.5*cm, width=7*cm, height=7.5*cm, preserveAspectRatio=True)

    c.showPage()


    # Side 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, H - margin, "Diagnostik & rumlig trend")

    if os.path.exists(fig_paths["rc"]):
        c.drawImage(fig_paths["rc"], margin, H - margin - 12*cm,
                    width=10*cm, height=10*cm, preserveAspectRatio=True)

    if "map" in fig_paths and fig_paths["map"] and os.path.exists(fig_paths["map"]):
        c.drawImage(fig_paths["map"], margin+10.5*cm, H - margin - 12*cm,
                    width=8*cm, height=10*cm, preserveAspectRatio=True)

    # --- INDSÆT INDEKSGRAF HER (NYT) ---
    if "index" in fig_paths and fig_paths["index"] and os.path.exists(fig_paths["index"]):
        c.drawImage(fig_paths["index"], margin, 3.0*cm,
                    width=16*cm, height=7.0*cm, preserveAspectRatio=True)

    c.setFont("Helvetica", 8)
    c.drawString(margin, 1.4*cm,
        "Metode: DML trend (to-komponent abundance, propensity score, residual confounding). Fink et al. 2023, Methods in Ecology and Evolution.")
    c.save()



# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output)
    figs_dir = os.path.join(output_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # --- Mode: save_preprocessed ---
    if args.mode == "save_preprocessed":
        print("Indlæser data parallelt...")
        df = read_parquet_parallel(
            args.data_dir, args.pattern, args.workers,
            start_year=args.start_year, end_year=args.end_year
        )
        print(f"Rå rækker: {len(df):,}")
        print("Feature-engineering (DOF-format, decimalkomma, tider)...")
        df = engineer_features_parallel(df, workers=args.workers)
        print("Projekterer til LAEA og tildeler 5x5 km grid...")
        df = project_to_laea(df, args.grid_crs)
        df = assign_grid_cells(df, args.grid_size_km)
        print(f"Gemmer preprocessed data til: {args.preprocessed_path}")
        df.to_parquet(args.preprocessed_path)
        print("Preprocessed data gemt.")
        return

    # --- Mode: use_preprocessed ---
    if args.mode == "use_preprocessed":
        print(f"Indlæser preprocessed data fra: {args.preprocessed_path}")
        df = pd.read_parquet(args.preprocessed_path)
        print(f"Preprocessed rækker: {len(df):,}")
    else:
        # --- Mode: normal (default) ---
        print("Indlæser data parallelt...")
        df = read_parquet_parallel(
            args.data_dir, args.pattern, args.workers,
            start_year=args.start_year, end_year=args.end_year
        )
        print(f"Rå rækker: {len(df):,}")
        print("Feature-engineering (DOF-format, decimalkomma, tider)...")
        df = engineer_features_parallel(df, workers=args.workers)
        print("Filter art/år...")
        df = filter_by_species_year(df, args.species_field, args.species, args.start_year, args.end_year)
        if len(df) == 0:
            raise ValueError("Ingen data efter filter (art/år/koordinater).")
        print("Projekterer til LAEA og tildeler 5x5 km grid...")
        df = project_to_laea(df, args.grid_crs)
        df = assign_grid_cells(df, args.grid_size_km)

    meta = {
        "species_field": args.species_field,
        "species": args.species if args.species else "",
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "n": f"{len(df):,}"
    }

    print("Estimerer DML trend (global + evt. heterogen)...")
    res = estimate_trend_components(df, heterogeneous=args.heterogeneous, ensemble_B=20)

    print("Kører artsspecifik residual confounding simulering og kalibrering...")
    beta0, beta1, sim_df = residual_confounding_via_simulation(df, reps_per_scenario=2)
    tau_adj_log = beta0 + beta1 * res["tau_abund_global_log"]
    ppy_adj = 100.0 * (np.exp(tau_adj_log) - 1.0)
    meta["beta0"] = beta0
    meta["beta1"] = beta1
    meta["PPY_RC"] = ppy_adj

    # --- NEW: Byg populationsindeks pr. år + bootstrap-CI pr. år ---
    years = np.arange(meta["year_min"], meta["year_max"] + 1)
    y0 = years.min()

    # central RC-justeret tau og indeks
    tau_c = tau_adj_log
    idx_c = 100.0 * np.exp(tau_c * (years - y0))

    # bootstrap-CI pr. år (RC-juster alle tau_samples)
    tau_b = res.get("tau_samples", None)
    if tau_b is not None and len(tau_b) > 0:
        tau_b_rc = beta0 + beta1 * np.array(tau_b)
        # matrix: hver række er et bootstrap, kolonner er år
        idx_mat = 100.0 * np.exp(np.outer(tau_b_rc, (years - y0)))
        idx_lo = np.percentile(idx_mat, 10, axis=0)
        idx_hi = np.percentile(idx_mat, 90, axis=0)
    else:
        # fallback: brug PPY-CI til at skabe approximative tau-lo/hi
        tau_lo = np.log1p(res["PPY_abundance_CI80_low"] / 100.0)
        tau_hi = np.log1p(res["PPY_abundance_CI80_high"] / 100.0)
        idx_lo = 100.0 * np.exp(tau_lo * (years - y0))
        idx_hi = 100.0 * np.exp(tau_hi * (years - y0))

    # Gem indeks CSV
    index_csv_path = os.path.splitext(args.output)[0] + "_index.csv"
    idx_df = pd.DataFrame({
        "year": years,
        "index": idx_c,
        "index_lo80": idx_lo,
        "index_hi80": idx_hi
    })
    idx_df.to_csv(index_csv_path, index=False)
    print(f"Populationsindeks CSV gemt: {index_csv_path}")

    # Gem indeks PNG (graf)
    index_png_path = os.path.join(figs_dir, "index.png")
    os.makedirs(os.path.dirname(index_png_path), exist_ok=True)  # <-- tilføj denne linje
    plt.figure(figsize=(7,4))
    plt.plot(years, idx_c, color="#2C7BB6", lw=2, label="Modelindeks (RC)")
    plt.fill_between(years, idx_lo, idx_hi, color="#2C7BB6", alpha=0.2, label="80% CI (bootstrap)")
    plt.axhline(100, color="k", lw=0.8, alpha=0.4)
    plt.xlabel("År"); plt.ylabel("Indeks (baseline = 100 i første år)")
    plt.title("Populationsudvikling (DML, RC-justeret)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(index_png_path, dpi=180); plt.close()
    print(f"Populationsindeks PNG gemt: {index_png_path}")

    out = {
        **meta,
        "PPY_abundance_global": res["PPY_abundance_global"],
        "PPY_abundance_CI80_low": res["PPY_abundance_CI80_low"],
        "PPY_abundance_CI80_high": res["PPY_abundance_CI80_high"],
        "PPY_abundance_global_RC": ppy_adj,
        "tau_abund_global_log": res["tau_abund_global_log"],
        "tau_abund_global_log_RC": tau_adj_log,
        "tau_det_global": res["tau_det_global"],
        "tau_cnt_global": res["tau_cnt_global"],
        "PPY_detection_global": res["PPY_detection_global"],
        "PPY_count_global": res["PPY_count_global"],
    }
    pd.DataFrame([out]).to_csv(args.output, index=False)
    print(f"Resultater gemt: {args.output}")
        
    # Figurer
    fig_paths = {}
    fig_paths["ts"]   = os.path.join(figs_dir, "ts.png")
    fig_paths["comp"] = os.path.join(figs_dir, "components.png")
    fig_paths["rc"]   = os.path.join(figs_dir, "rc_calibration.png")
    fig_paths["map"]  = None
    # Vi sætter IKKE fig_paths["index"] her; den peges senere til år-til-år figuren.

    print("Laver figurer...")
    plot_time_series(df, fig_paths["ts"])
    plot_components_bar(res["PPY_detection_global"], res["PPY_count_global"], fig_paths["comp"])
    plot_tau_calibration(sim_df, fig_paths["rc"])

    # Kort kun hvis heterogenitet er estimeret og tabellen findes
    if args.heterogeneous and ("tau_grid_table" in res):
        fig_paths["map"] = os.path.join(figs_dir, "map.png")
        plot_trend_map(res["tau_grid_table"], fig_paths["map"])

    # --- ÅR-TIL-ÅR: DML årseffekter + RC-justering ---
    yearwise_df, y0 = estimate_yearwise_components(df, lambda_diff=5.0)

    # RC på offsets (valgfrit at gemme/plotte til QA)
    yearwise_df = yearwise_df.sort_values("year").reset_index(drop=True)
    yearwise_df["beta_abund_rc"] = beta0 + beta1 * yearwise_df["beta_abund"]
    yearwise_df["index_rc"]      = 100.0 * np.exp(yearwise_df["beta_abund_rc"])

    # RC på slopes (anbefalet): baseline = 0 i første år
    tau_year    = yearwise_df["beta_abund"].diff().fillna(0.0)   # Δβ_y
    tau_year_rc = beta0 + beta1 * tau_year
    tau_year_rc.iloc[0] = 0.0                                    # baseline = 0
    beta_rc = tau_year_rc.cumsum()
    yearwise_df["beta_abund_rc_slope"] = beta_rc
    yearwise_df["index_rc_slope"]      = 100.0 * np.exp(beta_rc)

    # Stratificeret bootstrap pr. år -> 80% CI på slope-RC-kurven
    print("Bootstrap (år-stratificeret) for 80% CI...")
    years_ci, lo80, hi80 = bootstrap_yearwise(df, beta0=beta0, beta1=beta1, B=20, lambda_diff=5.0, random_state=123)

    # Gem CSV (inkl. CI-bånd for slope-RC)
    yearwise_df["index_rc_slope_lo80"] = lo80
    yearwise_df["index_rc_slope_hi80"] = hi80
    year_idx_csv = os.path.splitext(args.output)[0] + "_yearwise_index.csv"
    yearwise_df.to_csv(year_idx_csv, index=False)
    print(f"År-til-år indeks CSV gemt: {year_idx_csv}")

    # Plot år-indeks (PRIMÆRT: slope-RC + 80% CI)
    year_idx_slope_png = os.path.join(figs_dir, "index_yearwise_slope.png")
    plt.figure(figsize=(7,4))
    plt.plot(yearwise_df["year"], yearwise_df["index_rc_slope"],
            color="#0C7BDC", lw=2, label="Årsindeks (RC på slopes)")
    plt.scatter(yearwise_df["year"], yearwise_df["index_rc_slope"],
                color="#0C7BDC", s=16)
    plt.fill_between(yearwise_df["year"], yearwise_df["index_rc_slope_lo80"],
                    yearwise_df["index_rc_slope_hi80"], color="#0C7BDC", alpha=0.20,
                    label="80% CI (bootstrap)")
    
    # (valgfrit) vis også offset-RC som stiplet kurve
    plt.plot(yearwise_df["year"], yearwise_df["index_rc"],
            color="#7CB342", lw=1.5, ls="--", alpha=0.8,
            label="Årsindeks (RC på offsets)")
    plt.axhline(100, color="k", lw=0.8, alpha=0.4)
    plt.xlabel("År"); plt.ylabel("Indeks (baseline=100)")
    plt.title("År-til-år populationsindeks (DML, RC-justeret)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(year_idx_slope_png, dpi=180); plt.close()
    print(f"År-til-år indeks (slope-RC) PNG gemt: {year_idx_slope_png}")

    # Peg PDF'en til den nye år-til-år figur (slope-RC)
    fig_paths["index"] = year_idx_slope_png

    # PDF-rapport i output_dir
    pdf_path = args.report_pdf
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(output_dir, os.path.basename(args.report_pdf))
    print("Genererer PDF-rapport...")
    make_pdf_report(pdf_path, meta, res, sim_df, fig_paths)
    print(f"Rapport gemt: {pdf_path}")

if __name__ == "__main__":
    main()
