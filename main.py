#!/usr/bin/env python3
"""Minute-by-minute quantile forecaster for Polymarket's "Next CEO of NVIDIA?".

Entry for make.fun bounty polymarket-nvidia-ceo-forecast-01.

Method (per market, from the case's own minute prices only):
  1. Empirical quantiles of h-minute price changes over the last 48 hours,
     recency-weighted with a 12-hour half-life, computed on a grid of horizons
     and interpolated in sqrt(h). Prediction-market prices are sticky and
     zero-inflated, so the empirical change distribution is far better
     calibrated than any fixed-width normal band.
  2. Each empirical deviation is shrunk per quantile level (tails 0.60,
     inner 0.70, median 0.85) and blended 65/35 with the persistence
     baseline's normal deviation (sigma 0.0025 per sqrt-minute); the blended
     magnitude is clamped to [0.2, 2.0]x the baseline's, so a quiet or wild
     lookback can never produce an absurd band.
  3. A move-frequency regime scale: if the market moved in a larger (smaller)
     share of the last 3 hours' minutes than over its whole history, all
     deviations are widened (narrowed) by that ratio^0.25, clamped to
     [0.5, 1.5] — volatility clusters, and a market that just woke up tends
     to keep moving.
  4. Rows are clamped to [0, 1], rounded to 6 decimals and sorted, so every
     row is a valid non-decreasing quantile set. Fully deterministic:
     no randomness, no clock, no network, standard library only.
"""
import json
import math
import os
import sys
from statistics import NormalDist

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

HGRID = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 65, 80, 100, 125, 155, 195, 240]
LOOKBACK = 2880          # minutes of history used for the change distribution
WQ_HALFLIFE = 720.0      # recency half-life of a change, in minutes
BASE_SIGMA = 0.0025      # the house baseline's spread per sqrt-minute
BLEND_BASE = 0.35        # weight of the baseline deviation in the blend
LEVEL_SCALES = (0.60, 0.70, 0.85, 0.70, 0.60)  # per quantile level: 5/25/50/75/95
CLAMP_LO, CLAMP_HI = 0.2, 2.0
FREQ_SPAN = 180          # minutes used for the move-frequency regime
FREQ_POW = 0.25
FREQ_LO, FREQ_HI = 0.5, 1.5

_NORM = NormalDist()


def weighted_quantile(pairs, tau):
    """pairs: (value, weight) sorted by value; returns the weighted tau-quantile."""
    total = 0.0
    for _, w in pairs:
        total += w
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            return v
    return pairs[-1][0]


def change_curves(prices, taus):
    """Returns (hs, curves): for each grid horizon h in hs, the recency-weighted
    empirical quantiles of the h-minute change at the levels taus."""
    series = prices[-LOOKBACK:]
    m = len(series)
    lam = 0.5 ** (1.0 / WQ_HALFLIFE)
    curves = {}
    for h in HGRID:
        if h >= m:
            continue
        pairs = sorted((series[t] - series[t - h], lam ** (m - 1 - t))
                       for t in range(h, m))
        curves[h] = [weighted_quantile(pairs, tau) for tau in taus]
    return sorted(curves), curves


def interp_change(hs, curves, h):
    if h <= hs[0]:
        f = math.sqrt(h) / math.sqrt(hs[0])
        return [v * f for v in curves[hs[0]]]
    if h >= hs[-1]:
        return list(curves[hs[-1]])
    for i in range(len(hs) - 1):
        if hs[i] <= h <= hs[i + 1]:
            a, b = hs[i], hs[i + 1]
            f = (math.sqrt(h) - math.sqrt(a)) / (math.sqrt(b) - math.sqrt(a))
            return [curves[a][j] * (1 - f) + curves[b][j] * f for j in range(len(curves[a]))]
    return list(curves[hs[-1]])


def freq_scale(prices):
    n = len(prices)
    if n < 30:
        return 1.0
    moves_all = sum(1 for i in range(1, n) if prices[i] != prices[i - 1]) / (n - 1)
    if moves_all <= 0:
        return 1.0
    k = min(FREQ_SPAN, n - 1)
    moves_rec = sum(1 for i in range(n - k, n) if prices[i] != prices[i - 1]) / k
    sc = (moves_rec / moves_all) ** FREQ_POW
    return min(FREQ_HI, max(FREQ_LO, sc))


def price6(v):
    return round(min(1.0, max(0.0, v)) * 1e6) / 1e6 + 0.0


def forecast_market(prices, horizon, taus, zs):
    x = prices[-1]
    if len(prices) < 30:
        return [[price6(x + z * BASE_SIGMA * math.sqrt(h)) for z in zs]
                for h in range(1, horizon + 1)]
    hs, curves = change_curves(prices, taus)
    if not hs:
        return [[price6(x + z * BASE_SIGMA * math.sqrt(h)) for z in zs]
                for h in range(1, horizon + 1)]
    sc = freq_scale(prices)
    rows = []
    for h in range(1, horizon + 1):
        dq = interp_change(hs, curves, h)
        sqh = math.sqrt(h)
        row = []
        for j in range(len(taus)):
            emp_dev = dq[j] * LEVEL_SCALES[j]
            base_dev = zs[j] * BASE_SIGMA * sqh
            dev = (1 - BLEND_BASE) * emp_dev + BLEND_BASE * base_dev
            if base_dev != 0:
                mag = min(CLAMP_HI * abs(base_dev), max(CLAMP_LO * abs(base_dev), abs(dev)))
                dev = math.copysign(mag, dev if dev != 0 else base_dev)
            row.append(price6(x + dev * sc))
        row.sort()
        rows.append(row)
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for fname in files:
        case_id = fname[:-3]
        with open(os.path.join(INPUT_DIR, fname), encoding="utf8") as fh:
            case = json.load(fh)
        taus = case["quantiles"]
        zs = [_NORM.inv_cdf(t) for t in taus]
        horizon = int(case["horizon_minutes"])
        forecasts = []
        for mkt in case["markets"]:
            q = forecast_market(mkt["prices"], horizon, taus, zs)
            forecasts.append({"id": mkt["id"], "q": q})
        with open(os.path.join(OUTPUT_DIR, f"{case_id}.out"), "w", encoding="utf8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
