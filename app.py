"""
Market Regime Detector — Real-Time Streamlit App
Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import joblib
import os
from datetime import datetime, timedelta

import streamlit as st
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime Detector",
    page_icon="📈",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"]  { font-size: 1.6rem; font-weight: 600; }
  [data-testid="stMetricLabel"]  { font-size: 0.8rem; color: #888; }
  .regime-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.03em;
  }
  .bull     { background: #d4f4e8; color: #0f6e56; }
  .bear     { background: #fde8e8; color: #a32d2d; }
  .sideways { background: #ebebeb; color: #444; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TICKER         = "^GSPC"
TRAIN_START    = "2000-01-01"
TRAIN_END      = "2018-12-31"   # hold out 2019-present for live use
N_STATES       = 3
N_ITER         = 1000
N_FITS         = 10
MODEL_PATH     = "regime_model.pkl"
REFRESH_SECS   = 3600           # re-fetch data every hour

REGIME_COLORS  = {"bull": "#1D9E75", "bear": "#E24B4A", "sideways": "#888780"}
REGIME_EMOJI   = {"bull": "🟢", "bear": "🔴", "sideways": "⚪"}

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        140,
    "font.size":         10,
})

# ── Helper functions ───────────────────────────────────────────────────────────

def get_features(ticker: str, start: str, end: str | None = None) -> tuple[pd.Series, pd.DataFrame]:
    df    = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    close = df["Close"].squeeze()

    log_ret  = np.log(close / close.shift(1))
    vol_20   = log_ret.rolling(20).std()
    ma_ratio = close.rolling(50).mean() / close.rolling(200).mean() - 1

    feats = pd.DataFrame({
        "log_ret":  log_ret,
        "vol_20":   vol_20,
        "ma_ratio": ma_ratio,
    }).dropna()

    return close.loc[feats.index], feats


def train_model(features: pd.DataFrame) -> tuple[GaussianHMM, StandardScaler]:
    scaler = StandardScaler()
    X      = scaler.fit_transform(features.values)

    best_model, best_score = None, -np.inf
    for seed in range(N_FITS):
        m = GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=N_ITER,
            random_state=seed,
        )
        m.fit(X)
        s = m.score(X)
        if s > best_score:
            best_score, best_model = s, m

    return best_model, scaler


def label_regimes(
    model: GaussianHMM,
    scaler: StandardScaler,
    features: pd.DataFrame,
    close: pd.Series,
) -> pd.DataFrame:
    X      = scaler.transform(features.values)
    states = model.predict(X)

    mean_rets  = {s: features["log_ret"].values[states == s].mean() for s in range(N_STATES)}
    sorted_s   = sorted(mean_rets, key=mean_rets.get)
    label_map  = {sorted_s[0]: "bear", sorted_s[1]: "sideways", sorted_s[2]: "bull"}

    return pd.DataFrame({
        "close":  close,
        "state":  states,
        "regime": pd.Series(states, index=features.index).map(label_map),
    })


def compute_metrics(df: pd.DataFrame) -> dict:
    r     = np.log(df["close"] / df["close"].shift(1)).dropna()
    strat = r.copy()
    strat[df.loc[r.index, "regime"] != "bull"] = 0.0

    def sharpe(x): return (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
    def max_dd(x):
        c = (1 + x).cumprod()
        return (c / c.cummax() - 1).min()
    def cagr(x):
        n = len(x) / 252
        return (1 + x).prod() ** (1 / n) - 1

    return {
        "sharpe_strat":  sharpe(strat),
        "sharpe_bah":    sharpe(r),
        "maxdd_strat":   max_dd(strat),
        "maxdd_bah":     max_dd(r),
        "cagr_strat":    cagr(strat),
        "cagr_bah":      cagr(r),
        "cum_strat":     (1 + strat).cumprod(),
        "cum_bah":       (1 + r).cumprod(),
    }


def plot_regimes(df: pd.DataFrame, zoom_years: int | None = None) -> plt.Figure:
    plot_df = df.iloc[-zoom_years * 252:] if zoom_years else df

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 6),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    ax1.plot(plot_df.index, plot_df["close"], color="#378ADD", lw=0.9, zorder=3)

    prev, start = plot_df["regime"].iloc[0], plot_df.index[0]
    for date, row in plot_df.iterrows():
        if row["regime"] != prev:
            ax1.axvspan(start, date, alpha=0.18, color=REGIME_COLORS[prev], lw=0)
            start, prev = date, row["regime"]
    ax1.axvspan(start, plot_df.index[-1], alpha=0.18, color=REGIME_COLORS[prev], lw=0)

    patches = [mpatches.Patch(color=c, alpha=0.6, label=r.capitalize())
               for r, c in REGIME_COLORS.items()]
    ax1.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0)
    ax1.set_ylabel("SPX Price", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    strip_colors = plot_df["regime"].map(REGIME_COLORS)
    ax2.bar(plot_df.index, [1] * len(plot_df), color=strip_colors, width=2, align="center")
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["Regime"], fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    return fig


def plot_equity(metrics: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(metrics["cum_strat"].index, metrics["cum_strat"].values,
            color="#1D9E75", lw=1.2, label="Regime strategy")
    ax.plot(metrics["cum_bah"].index,   metrics["cum_bah"].values,
            color="#378ADD", lw=0.9, label="Buy & hold", alpha=0.8)
    ax.set_ylabel("Growth of $1", fontsize=10)
    ax.legend(fontsize=9, framealpha=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    return fig


# ── Cached data & model (refreshes every REFRESH_SECS seconds) ────────────────

@st.cache_data(ttl=REFRESH_SECS, show_spinner=False)
def load_live_data() -> tuple[pd.Series, pd.DataFrame]:
    """Fetch the latest SPX data up to today."""
    return get_features(TICKER, TRAIN_START)


@st.cache_resource(show_spinner=False)
def load_or_train_model(train_end: str = TRAIN_END):
    """
    Load a saved model if it exists, otherwise train on TRAIN_START → train_end.
    Uses st.cache_resource so the model lives in memory across reruns.
    """
    if os.path.exists(MODEL_PATH):
        saved  = joblib.load(MODEL_PATH)
        return saved["model"], saved["scaler"]

    _, feats_train = get_features(TICKER, TRAIN_START, train_end)
    model, scaler  = train_model(feats_train)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
    return model, scaler


# ── App layout ─────────────────────────────────────────────────────────────────

st.title("📈 Market Regime Detector")
st.caption("S&P 500 · Hidden Markov Model · 3 regimes: bull / sideways / bear")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    zoom_options = {"All data": None, "Last 10 years": 10, "Last 5 years": 5, "Last 2 years": 2}
    zoom_label   = st.selectbox("Chart window", list(zoom_options.keys()), index=0)
    zoom_years   = zoom_options[zoom_label]

    retrain      = st.button("🔄 Retrain model", help="Delete saved model and retrain from scratch")
    if retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.success("Model cleared — will retrain on next load.")

    st.divider()
    st.markdown("**Data**")
    st.caption(f"Train window: {TRAIN_START} → {TRAIN_END}")
    st.caption(f"Live data: {TRAIN_END} → today")
    st.caption(f"Cache refresh: every {REFRESH_SECS // 60} min")

# Load data and model
with st.spinner("Fetching latest SPX data…"):
    close_all, features_all = load_live_data()

with st.spinner("Loading model (first run trains from scratch, ~30s)…"):
    model, scaler = load_or_train_model()

# Label regimes on ALL data (including post-training live data)
regime_df = label_regimes(model, scaler, features_all, close_all)
metrics   = compute_metrics(regime_df)

# ── Current regime banner ──────────────────────────────────────────────────────
latest         = regime_df.iloc[-1]
current_regime = latest["regime"]
current_price  = latest["close"]
prev_close     = regime_df["close"].iloc[-2]
day_chg        = (current_price / prev_close - 1) * 100

badge_class = current_regime  # bull / bear / sideways CSS class

col_badge, col_price, col_date = st.columns([2, 1, 1])
with col_badge:
    st.markdown(
        f"Current regime &nbsp;"
        f'<span class="regime-badge {badge_class}">'
        f'{REGIME_EMOJI[current_regime]} {current_regime.upper()}'
        f"</span>",
        unsafe_allow_html=True,
    )
with col_price:
    st.metric("SPX Price", f"{current_price:,.0f}", f"{day_chg:+.2f}% today")
with col_date:
    st.metric("As of", regime_df.index[-1].strftime("%d %b %Y"))

st.divider()

# ── Regime distribution ────────────────────────────────────────────────────────
dist = regime_df["regime"].value_counts(normalize=True).mul(100)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🟢 Bull %",     f"{dist.get('bull',0):.1f}%")
c2.metric("⚪ Sideways %", f"{dist.get('sideways',0):.1f}%")
c3.metric("🔴 Bear %",     f"{dist.get('bear',0):.1f}%")
c4.metric("Trading days",  f"{len(regime_df):,}")

# ── Charts ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Regime overlay", "Equity curves"])

with tab1:
    st.pyplot(plot_regimes(regime_df, zoom_years=zoom_years), use_container_width=True)

with tab2:
    st.pyplot(plot_equity(metrics), use_container_width=True)

# ── Backtest metrics ───────────────────────────────────────────────────────────
st.subheader("Backtest summary")
st.caption("⚠️ Model trained on 2000–2018. Metrics shown on full history (includes out-of-sample 2019–present).")

m1, m2, m3 = st.columns(3)
m1.metric("Sharpe — strategy",  f"{metrics['sharpe_strat']:.2f}",
          f"{metrics['sharpe_strat'] - metrics['sharpe_bah']:+.2f} vs B&H")
m2.metric("Max drawdown — strategy", f"{metrics['maxdd_strat']*100:.1f}%",
          f"{(metrics['maxdd_strat'] - metrics['maxdd_bah'])*100:+.1f}% vs B&H")
m3.metric("CAGR — strategy",    f"{metrics['cagr_strat']*100:.1f}%",
          f"{(metrics['cagr_strat'] - metrics['cagr_bah'])*100:+.1f}% vs B&H")

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("Show raw regime data (last 30 days)"):
    st.dataframe(
        regime_df.tail(30)[["close", "regime"]]
        .rename(columns={"close": "SPX Close", "regime": "Regime"})
        .sort_index(ascending=False)
        .style.applymap(
            lambda v: f"color: {REGIME_COLORS.get(v, '#000')}; font-weight: 600",
            subset=["Regime"],
        ),
        use_container_width=True,
    )

st.divider()
st.caption(f"Data via yfinance · Model: Gaussian HMM ({N_STATES} states, full covariance) · "
           f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}")
