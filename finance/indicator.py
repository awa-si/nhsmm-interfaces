import os
import re
import time
import json
import logging
import numpy as np
import polars as pl
import scipy.signal
from types import SimpleNamespace

import talib.abstract as ta
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "/opt/trader/user_data/data/bybit/futures"


class Strategy: # dummy spacename
    timeframe = "5m"
    informative_tfs = ["15m", "1h", "1d"]
    dp = None

class Indicator:

    ohlcv = ("high", "low", "open", "close", "volume")

    FEATURE_PATTERNS = {
        "ohlcv": [r"^(open|high|low|close|volume)(_?\d+[smhdw])?$"],
        "trend": [
            r"^aroon_(up|down)(_?\d+[smhdw])?$",
            r"^adx(_\d+[smhdw])?$",
            r"^trix(_\d+[smhdw])?$",
            r"^ppo(_\d+[smhdw])?$",
            r"^\+di(_\d+[smhdw])?$",
            r"^-di(_\d+[smhdw])?$",
            r"^ema_(fast|mid|slow)(_?\d+[smhdw])?$",
            r"^returns$",
        ],
        "bands": [
            r"^bb_(upper|middle|lower)$",
            r"^bb_width(_\d+[smhdw])?$",
        ],
        "volatility": [
            r"^atr(_(pct|ratio|\d+[smhdw]))?$",
            r"^hv(\_\d+)?$",
        ],
        "derived": [r"^derived\..*"],
        "divergences": [
            r"^[a-zA-Z0-9_]+_divergence(\..+)?$",
            r"^divergences\..*"
        ],
        "volume": [
            r"^(obv|volume_ma.*)$",
            r"^volume_metrics\..*",
        ],
        "oscillator": [
            r"^rsi(_(fast|mid|slow))?(_\d+[smhdw])?$",
            r"^stoch(K|D)(_?\d+[smhdw])?$",
            r"^williams_r(_\d+[smhdw])?$",
            r"^mfi(_\d+[smhdw])?$",
        ],
        "momentum": [
            r"^roc(_(fast|mid|slow))?(_\d+[smhdw])?$",
            r"^cci(_\d+[smhdw])?$",
        ],
        "macro": [
            r"^macd(_(signal|hist))?$",
            r"^macd$",
        ],
        "structure": [
            r"^fibs.*",
            r"^swings.*",
            r"^gap\..*",
            r"^fvg\..*",
        ],
        "session": [r"^daily\..*"],
    }

    def __init__(self, strategy: Strategy = None, params: dict = None):
        default = {
            "adx": 10,
            "cci": 20,
            "mfi": 14,
            "aroon": 25,
            "trix": 15,
            "williams_r": 14,
            "atr": 14, "atr_ratio": 20,
            "bb_dev": 2.0, "bb_period": 20,
            "ema_fast": 10, "ema_mid": 100, "ema_slow": 200,
            "macd_fast": 6, "macd_slow": 26, "macd_signal": 9,
            "rsi_fast": 5, "rsi_mid": 9, "rsi_slow": 14,
            "roc_fast": 3, "roc_mid": 5, "roc_slow": 10,
            "stoch_fast": 14, "stoch_slow_d": 3, "stoch_slow_k": 3,
            "swing_window": 50, "vol_sma": 20,
            "vpoc_bins": 50, "vpoc_lookback": 72,
            "vwap_lookback": 50,
            "shift_policy": {},
        }
        self.params = SimpleNamespace(**{**default, **(params or {})})
        self._param_space: dict[str, tuple] = {}
        self.strategy = strategy
        self.registry = None

    def _build_registry(self, df: pl.DataFrame, extra_patterns=None) -> dict:
        compiled = {
            g: [re.compile(p) for p in patterns]
            for g, patterns in (self.FEATURE_PATTERNS | (extra_patterns or {})).items()
        }

        registry = {
            "ohlcv": {},
            "bands": {},
            "divergences": {},
            "derived": {},
            "macro": {},
            "momentum": {},
            "oscillator": {},
            "session": {},
            "structure": {},
            "trend": {},
            "volatility": {},
            "volume": {},
            "unknown": [],
            "ambiguous": {},
            "base": self.strategy.timeframe,
        }

        def add_unique(bucket: dict, tf: str, feat: str):
            bucket.setdefault(tf, [])
            if feat not in bucket[tf]:
                bucket[tf].append(feat)

        def add_unknown(name: str):
            if name not in registry["unknown"]:
                registry["unknown"].append(name)

        def match_group(name: str) -> str:
            matches = []
            for g, regexes in compiled.items():
                if any(r.fullmatch(name) or r.match(name) for r in regexes):
                    matches.append(g)
            if not matches:
                return "unknown"
            if len(matches) > 1:
                registry["ambiguous"][name] = matches
                return "unknown"
            return matches[0]

        for col in df.columns:
            # --- detect timeframe + base ---
            m = re.search(r"_(\d+[smhdw])$", col)
            if m:
                tf = m.group(1)
                base = col[: -len(m.group(0))]
            else:
                tf = getattr(self, "strategy", None) and self.strategy.timeframe or "base"
                base = col

            # --- divergences ---
            if col.startswith("divergences") and df[col].dtype == pl.Struct:
                for sub, field_dtype in df[col].dtype.to_schema().items():
                    if isinstance(field_dtype, pl.Struct):
                        for f in field_dtype.to_schema().keys():
                            canon = f"divergences.{sub}.{f}"
                            add_unique(registry["divergences"], tf, canon)
                    else:
                        canon = f"divergences.{sub}"
                        add_unique(registry["divergences"], tf, canon)
                continue

            # --- volume features ---
            if col.startswith("volume_metrics") and df[col].dtype == pl.Struct:
                for sub, field_dtype in df[col].dtype.to_schema().items():
                    if isinstance(field_dtype, pl.Struct):
                        for f in field_dtype.to_schema().keys():
                            canon = f"volume_metrics.{sub}.{f}"
                            add_unique(registry["volume"], tf, canon)
                    else:
                        canon = f"volume_metrics.{sub}"
                        add_unique(registry["volume"], tf, canon)
                continue

            # --- derived features ---
            if col.startswith("derived") and df[col].dtype == pl.Struct:
                for sub, field_dtype in df[col].dtype.to_schema().items():
                    if isinstance(field_dtype, pl.Struct):
                        for f in field_dtype.to_schema().keys():
                            canon = f"derived.{sub}.{f}"
                            add_unique(registry["derived"], tf, canon)
                    else:
                        canon = f"derived.{sub}"
                        add_unique(registry["derived"], tf, canon)
                continue

            # --- daily/session struct features ---
            if col.startswith("daily") and df[col].dtype == pl.Struct:
                for sub, field_dtype in df[col].dtype.to_schema().items():
                    if isinstance(field_dtype, pl.Struct):
                        for f in field_dtype.to_schema().keys():
                            canon = f"daily.{sub}.{f}"
                            add_unique(registry["session"], tf, canon)
                    else:
                        canon = f"daily.{sub}"
                        add_unique(registry["session"], tf, canon)
                continue

            # --- other struct expansion ---
            if df[col].dtype == pl.Struct:
                for f in df[col].dtype.to_schema().keys():
                    full_path = f"{base}.{f}"
                    g = match_group(full_path) or match_group(f)
                    if g == "unknown":
                        add_unknown(full_path)
                    else:
                        add_unique(registry[g], tf, full_path)
                continue

            # --- scalar feature ---
            g = match_group(base)
            if g == "unknown":
                add_unknown(base)
            else:
                add_unique(registry[g], tf, base)

        return registry

    @property
    def max_lookback(self) -> int:
        keys = [
            k for k in vars(self.params) if any(s in k for s in ("lookback", "period", "window", "fast", "slow"))
        ]
        lookbacks = [
            getattr(self.params, k)
            for k in keys if isinstance(getattr(self.params, k), int) and getattr(self.params, k) > 1
        ]
        return max(lookbacks)

    @staticmethod
    def _ohlcv(df: pl.DataFrame) -> dict[str, np.ndarray]:
        return {k: df[k].cast(pl.Float64).to_numpy() for k in Indicator.ohlcv}

    @staticmethod
    def _ta(df: pl.DataFrame, name: str = None, func=None, *args, length: int = None, names: list[str] = None, **kwargs):

        if name is None and func is None:
            return Indicator._ohlcv(df)

        if length is None: length = len(df)
        ohlcv = Indicator._ohlcv(df)

        mapped_args = [ohlcv[a] if isinstance(a, str) and a in ohlcv else a for a in args]
        arr = func(*mapped_args, **kwargs)

        def pad(a):
            a = np.array(a, dtype=np.float64, copy=False)
            if a.shape[0] < length:
                a = np.pad(a, (length - a.shape[0], 0), constant_values=np.nan)
            return a

        if isinstance(arr, (tuple, list)):
            if names and len(names) == len(arr):
                colnames = names
            else:
                colnames = [f"{name}_{i}" for i in range(len(arr))]
            return [pl.Series(col, pad(a)) for col, a in zip(colnames, arr)]
        return pl.Series(name, pad(arr))

    def _load_dataframe(self, pair: str, timeframe: str) -> pl.DataFrame:
        if self.strategy.dp:
            df = self.strategy.dp.get_pair_dataframe(pair, timeframe)
            return pl.from_pandas(df).sort("date")
        symbol = pair.replace("/", "_").replace(":", "_")
        filename = f"{symbol}-{timeframe}-futures.feather"
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Freqtrade data not found for {pair} @ {timeframe}: {path}")
        return pl.read_ipc(path, memory_map=False).sort("date")

    def process(self, metadata: dict, use_htf: bool = True, drop_htf_ohlcv: bool = True) -> pl.DataFrame:
        pair = metadata.get("pair")
        df = self._load_dataframe(pair, self.strategy.timeframe)
        if df.is_empty(): return df

        # --- Main timeframe processing ---
        df = self.process_main_tf(df)
        if not self.strategy or not use_htf:
            return df

        logger.info("Processing higher timeframes...")
        for tf in self.strategy.informative_tfs:
            if tf == self.strategy.timeframe: continue
            inf = self._load_dataframe(pair, tf)

            atr = self._ta(inf, "atr", ta.ATR, "high", "low", "close", timeperiod=self.params.atr)
            inf = inf.with_columns([
                atr,
                (atr / (inf["close"].cast(pl.Float64) + 1e-9)).alias("atr_pct"),
                (atr / (atr.rolling_mean(self.params.atr_ratio) + 1e-9)).alias("atr_ratio"),
                self._ta(inf, "adx", ta.ADX, "high", "low", "close", timeperiod=self.params.adx),
                self._ta(inf, "+di", ta.PLUS_DI, "high", "low", "close", timeperiod=self.params.adx),
                self._ta(inf, "-di", ta.MINUS_DI, "high", "low", "close", timeperiod=self.params.adx),
                self._ta(inf, "ema_fast", ta.EMA, "close", timeperiod=self.params.ema_fast),
                self._ta(inf, "ema_mid", ta.EMA, "close", timeperiod=self.params.ema_mid),
                self._ta(inf, "ema_slow", ta.EMA, "close", timeperiod=self.params.ema_slow),
                self._ta(inf, "roc_fast", ta.ROC, "close", timeperiod=self.params.roc_fast),
                self._ta(inf, "roc_slow", ta.ROC, "close", timeperiod=self.params.roc_slow),
                self._ta(inf, "rsi_fast", ta.RSI, "close", timeperiod=self.params.rsi_fast),
                self._ta(inf, "rsi_slow", ta.RSI, "close", timeperiod=self.params.rsi_slow),
                self._ta(inf, "obv", ta.OBV, "close", "volume"),

                *self._ta(inf, None, ta.AROON, "high", "low", self.params.aroon, names=["aroon_up", "aroon_down"]),
                self._ta(inf, "trix", ta.TRIX, "close", self.params.trix),
                self._ta(inf, "ppo", ta.PPO, "close", 12, 26, 0),

                self._ta(inf, "cci", ta.CCI, "high", "low", "close", timeperiod=self.params.cci),
                self._ta(inf, "mfi", ta.MFI, "high", "low", "close", "volume", timeperiod=self.params.mfi),
                self._ta(inf, "williams_r", ta.WILLR, "high", "low", "close", timeperiod=self.params.williams_r),
            ])
            inf = self._htf_gap(inf, tf)
            inf = self.swing_high_low(inf, self.params.swing_window)

            inf = inf.rename({c: f"{c}_{tf}" for c in inf.columns if c != "date"})
            df = df.join_asof(inf, on="date", strategy="backward")

            # Forward fill indicators
            core = self.ohlcv + ("date",)
            ohlcv_cols = [
                c for c in df.columns
                if any(c.startswith(f"{p}_") for p in self.ohlcv)
            ]
            ind_cols = [c for c in df.columns if c not in core and c not in ohlcv_cols]
            df = df.with_columns([df[c].fill_null(strategy="forward") for c in ind_cols])

            # Apply shift policy
            if self.params.shift_policy and self.params.shift_policy.get(tf, False):
                df = df.with_columns([df[c].shift(1) for c in ind_cols])

            # Optionally drop HTF raw OHLCV
            if drop_htf_ohlcv:
                df = df.drop(ohlcv_cols)

        df = self.swing_high_low(df, self.params.swing_window)
        df = self._volume_features(df)
        df = df.with_columns([
            pl.struct([
                self._calc_divergence(df, col="rsi_slow"),
                self._calc_divergence(df, col="stochK"),
                self._calc_divergence(df, col="macd"),
            ]).alias("divergences")
        ])
        df = self._calc_derived(df)
        df = self._calc_fibs(df)
        df = self._daily(df)
        df = self._fvg(df)
        # print(df)

        # df[:500].write_ndjson("indicator_output.json")
        self.registry = self._build_registry(df)
        return df

    def process_main_tf(self, df: pl.DataFrame) -> pl.DataFrame:

        atr = self._ta(df, "atr", ta.ATR, "high", "low", "close", timeperiod=self.params.atr)
        bb = self._ta(
            df, "bb", ta.BBANDS, "close",
            timeperiod=self.params.bb_period,
            nbdevup=self.params.bb_dev,
            nbdevdn=self.params.bb_dev,
            names=["bb_upper", "bb_middle", "bb_lower"]
        )
        df = df.with_columns([
            self._ta(df, "adx", ta.ADX, "high", "low", "close", timeperiod=self.params.adx),
            atr,
            (atr / (df["close"].cast(pl.Float64) + 1e-9)).alias("atr_pct"),
            (atr / (atr.rolling_mean(self.params.atr_ratio) + 1e-9)).alias("atr_ratio"),

            self._ta(df, "+di", ta.PLUS_DI, "high", "low", "close", timeperiod=self.params.adx),
            self._ta(df, "-di", ta.MINUS_DI, "high", "low", "close", timeperiod=self.params.adx),

            self._ta(df, "ema_fast", ta.EMA, "close", timeperiod=self.params.ema_fast),
            self._ta(df, "ema_mid", ta.EMA, "close", timeperiod=self.params.ema_mid),
            self._ta(df, "ema_slow", ta.EMA, "close", timeperiod=self.params.ema_slow),

            self._ta(df, "obv", ta.OBV, "close", "volume"),
            df["volume"].rolling_mean(self.params.vol_sma).alias("volume_ma"),

            *bb,
            ((bb[0] - bb[2]) / (bb[1] + 1e-9)).alias("bb_width"),
            *(self._ta(
                df, "macd", ta.MACD, "close",
                fastperiod=self.params.macd_fast,
                slowperiod=self.params.macd_slow,
                signalperiod=self.params.macd_signal,
                names=["macd", "macd_signal", "macd_hist"]
            )),

            self._ta(df, "roc_fast", ta.ROC, "close", timeperiod=self.params.roc_fast),
            self._ta(df, "roc_slow", ta.ROC, "close", timeperiod=self.params.roc_slow),
            self._ta(df, "rsi_fast", ta.RSI, "close", timeperiod=self.params.rsi_fast),
            self._ta(df, "rsi_mid", ta.RSI, "close", timeperiod=self.params.rsi_mid),
            self._ta(df, "rsi_slow", ta.RSI, "close", timeperiod=self.params.rsi_slow),

            *self._ta(df, None, ta.AROON, "high", "low", self.params.aroon, names=["aroon_up", "aroon_down"]),
            self._ta(df, "trix", ta.TRIX, "close", self.params.trix),
            self._ta(df, "ppo", ta.PPO, "close", 12, 26, 0),

            self._ta(df, "cci", ta.CCI, "high", "low", "close", timeperiod=self.params.cci),
            self._ta(df, "mfi", ta.MFI, "high", "low", "close", "volume", timeperiod=self.params.mfi),
            self._ta(df, "williams_r", ta.WILLR, "high", "low", "close", timeperiod=self.params.williams_r),
            self._hv_proxy(df, window=50),

            *(self._ta(
                df, "stoch", ta.STOCH, "high", "low", "close",
                fastk_period=self.params.stoch_fast,
                slowk_period=self.params.stoch_slow_k,
                slowk_matype=0,
                slowd_period=self.params.stoch_slow_d,
                slowd_matype=0,
                names=["stochK", "stochD"]
            )),
        ])
        return df

    @staticmethod
    def _calc_divergence(df: pl.DataFrame, col: str = "rsi", order: int = 3, tol: int = 7, eps: float = 1.0, smooth: int = 3) -> pl.DataFrame:
        """
        Compute oscillator divergences with relaxed detection.
        Returns:
          {col}: struct with fields:
            - code: -2, -1, 0, +1, +2
            - conf: smoothed abs(z-score) (0 if no divergence)
            - z: raw signed z-score
        """

        base_col = "close"
        osc = df[col].to_numpy()
        price = df[base_col].to_numpy()
        n = len(df)

        if n < 2 * order + 1:
            return df.with_columns([
                pl.struct([
                    pl.Series(np.zeros(n, dtype=np.int8)).alias("type"),
                    pl.Series(np.zeros(n, dtype=np.float32)).alias("conf"),
                    pl.Series(np.zeros(n, dtype=np.float32)).alias("z"),
                ]).alias(f"{col}_div")
            ])

        window_size = 2 * order + 1
        window = sliding_window_view(price, window_size)

        is_local_min = (window[:, order] <= window[:, :order].min(axis=1)) & (window[:, order] <= window[:, order+1:].min(axis=1))
        is_local_max = (window[:, order] >= window[:, :order].max(axis=1)) & (window[:, order] >= window[:, order+1:].max(axis=1))

        min_idx = np.where(is_local_min)[0] + order
        max_idx = np.where(is_local_max)[0] + order

        div_code = np.zeros(n, dtype=np.int8)
        div_conf = np.zeros(n, dtype=np.float32)
        div_z = np.zeros(n, dtype=np.float32)

        def process_div(idx_arr, cmp_price, cmp_osc, score: int):
            nonlocal div_code, div_conf, div_z
            if len(idx_arr) < 2:
                return
            prev_idx, curr_idx = idx_arr[:-1], idx_arr[1:]
            valid = np.abs(curr_idx - prev_idx) <= tol
            prev_idx, curr_idx = prev_idx[valid], curr_idx[valid]

            for pi, ci in zip(prev_idx, curr_idx):
                price_ok = cmp_price(price[ci], price[pi])
                osc_diff = osc[ci] - osc[pi]
                osc_ok = cmp_osc(osc[ci], osc[pi]) or abs(osc_diff) < eps
                if price_ok and osc_ok:
                    div_code[ci] = score
                    seg = osc[max(0, pi - order):ci + 1]
                    mean = np.nanmean(seg)
                    std = np.nanstd(seg) + 1e-9
                    z = (osc_diff - mean) / std
                    div_z[ci] = float(z)
                    div_conf[ci] = abs(float(z))

        # bullish/bearish, regular/hidden
        process_div(min_idx, np.less, np.greater, +1) # bullish regular
        process_div(min_idx, np.greater, np.less, +2) # bullish hidden
        process_div(max_idx, np.greater, np.less, -1) # bearish regular
        process_div(max_idx, np.less, np.greater, -2) # bearish hidden

        # --- smoothing confidence ---
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            smoothed = np.convolve(div_conf, kernel, mode="same")
            div_conf = np.where(div_code != 0, smoothed, 0.0)

        return pl.struct([
            pl.Series(div_code).alias("type"),
            pl.Series(div_conf).alias("conf"),
            pl.Series(div_z).alias("z"),
        ]).alias(f"{col}")

    @staticmethod
    def _volume_features(df: pl.DataFrame, n_bins: int = 20, window: int = 50, atr_mult: float = 1.0, absorption_clip: float = 10.0) -> pl.DataFrame:

        va_pct: float = 0.7
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        atr = df["atr"].to_numpy() if "atr" in df.columns else np.ones(len(close))
        n = len(close)

        # --- Dynamic window selection ---
        if isinstance(window, (tuple, list)):
            min_w, max_w = window
            atr_med = np.nanmedian(atr)
            atr_now = atr[-1] if not np.isnan(atr[-1]) else atr_med
            scale = (atr_now / (atr_med + 1e-9)) * atr_mult
            dyn_window = int(np.clip(int((min_w + max_w) / 2 * scale), min_w, max_w))
        else:
            dyn_window = window

        # --- Allocate arrays ---
        vpoc_full = np.full(n, np.nan, dtype=np.float32)
        val_full = np.full(n, np.nan, dtype=np.float32)
        vah_full = np.full(n, np.nan, dtype=np.float32)

        if n >= dyn_window:
            close_win = sliding_window_view(close, dyn_window)
            vol_win = sliding_window_view(volume, dyn_window)

            lo = close_win.min(axis=1)
            hi = close_win.max(axis=1)

            # Bin prices
            norm_close = (close_win - lo[:, None]) / (hi - lo)[:, None]
            bin_idx = np.floor(norm_close * n_bins).astype(int)
            bin_idx[bin_idx == n_bins] = n_bins - 1

            hist = np.zeros((len(close_win), n_bins))
            for i in range(n_bins):
                hist[:, i] = (vol_win * (bin_idx == i)).sum(axis=1)

            # VPOC
            max_bin = hist.argmax(axis=1)
            vpoc_full[dyn_window - 1:] = lo + (hi - lo) * (max_bin + 0.5) / n_bins

            # Value area
            sorted_idx = np.argsort(-hist, axis=1)
            cum_vol = np.cumsum(np.take_along_axis(hist, sorted_idx, axis=1), axis=1)
            total_vol = cum_vol[:, -1][:, None]
            mask_va = cum_vol <= total_vol * va_pct
            val_bins = np.argmax(mask_va, axis=1)
            vah_bins = n_bins - 1 - np.argmax(np.flip(mask_va, axis=1), axis=1)

            val_full[dyn_window - 1:] = lo + (hi - lo) * val_bins / n_bins
            vah_full[dyn_window - 1:] = lo + (hi - lo) * (vah_bins + 1) / n_bins

        # --- Derived metrics ---
        va_width = vah_full - val_full
        va_width_atr = va_width / (atr + 1e-9)

        vwap_raw = np.cumsum(close * volume) / (np.cumsum(volume) + 1e-9)
        vwap = ta.WMA(vwap_raw, timeperiod=dyn_window)
        vwap = np.nan_to_num(vwap, nan=0.0)
        vwap_dev = (close - vwap) / (atr + 1e-9)

        delta = np.sign(np.diff(close, prepend=close[0]))
        vol_delta = delta * volume
        cvd = np.cumsum(vol_delta)
        cvd_norm = cvd / (atr + 1e-9)

        price_diff = np.abs(np.diff(close, prepend=close[0]))
        absorption = (volume / (price_diff + 1e-9)) / (atr + 1e-9)
        absorption = np.clip(absorption, -absorption_clip, absorption_clip)
        absorption_conf = np.nan_to_num(
            (absorption - np.nanmean(absorption)) / (np.nanstd(absorption) + 1e-9), nan=0.0
        )

        ret = np.diff(close, prepend=close[0])
        imbalance = (ret * volume) / (atr + 1e-9)
        imb_mean = ta.SMA(imbalance, timeperiod=dyn_window)
        imb_std = ta.STDDEV(imbalance, timeperiod=dyn_window)
        imbalance_z = np.nan_to_num((imbalance - imb_mean) / (imb_std + 1e-9), nan=0.0)

        buy_vol = np.where(delta > 0, volume, 0.0)
        sell_vol = np.where(delta < 0, volume, 0.0)
        raw_aggr = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)
        raw_aggr /= (atr + 1e-9)
        aggression = np.nan_to_num(ta.SMA(raw_aggr, timeperiod=dyn_window), nan=0.0)
        aggression_conf = np.nan_to_num(
            (aggression - np.nanmean(aggression)) / (np.nanstd(aggression) + 1e-9), nan=0.0
        )

        # === Struct output ===
        return df.with_columns([
            pl.struct([
                pl.struct([
                    pl.Series("low", val_full),
                    pl.Series("high", vah_full),
                    pl.Series("poc", vpoc_full),
                    pl.Series("width", va_width),
                    pl.Series("conf", va_width_atr),
                ]).alias("va"),
                pl.struct([
                    pl.Series("value", vwap),
                    pl.Series("conf", vwap_dev),
                ]).alias("vwap"),
                pl.struct([
                    pl.Series("value", cvd.astype(np.float32)),
                    pl.Series("conf", cvd_norm.astype(np.float32)),
                ]).alias("cvd"),
                pl.struct([
                    pl.Series("value", absorption.astype(np.float32)),
                    pl.Series("conf", absorption_conf.astype(np.float32)),
                ]).alias("absorption"),
                pl.struct([
                    pl.Series("value", imbalance.astype(np.float32)),
                    pl.Series("conf", imbalance_z.astype(np.float32)),
                ]).alias("imbalance"),
                pl.struct([
                    pl.Series("value", aggression.astype(np.float32)),
                    pl.Series("conf", aggression_conf.astype(np.float32)),
                ]).alias("aggression"),
            ]).alias("volume_metrics")
        ])

    def _calc_derived(self, df: pl.DataFrame) -> pl.DataFrame:
        roll_base = getattr(self.params, "rolling_window", 5)
        roll_short, roll_long = max(3, int(roll_base * 0.5)), max(5, int(roll_base * 1.5))
        returns = (df["close"] / df["close"].shift(1) - 1).fill_null(0.0)
        returns_roc = (df["roc_fast"] / 100).fill_null(0.0)
        return df.with_columns([
            returns.alias("returns"),
            pl.struct([
                # --- Returns / Momentum ---
                pl.struct([
                    (returns + 1).log().alias("log"),
                    self._ta(df, "momentum_20", ta.ROC, df["close"], timeperiod=20),
                    self._ta(df, "momentum_50", ta.ROC, df["close"], timeperiod=50),
                    self._ta(df, "rolling_ret_short", ta.EMA, returns, timeperiod=roll_short),
                    self._ta(df, "rolling_ret_long", ta.EMA, returns, timeperiod=roll_long),
                ]).alias("returns"),

                # --- Volatility / Regime ---
                pl.struct([
                    ((df["atr_pct"] - self._ta(df, "atr_pct_ma", ta.SMA, df["atr_pct"], 20)) /
                     (self._ta(df, "atr_pct_std", ta.STDDEV, df["atr_pct"], 20) + 1e-9)).alias("regime"),
                    self._ta(df, "clustering", ta.SMA, df["atr_pct"] * df["atr_pct"].shift(1), 5),
                    self._ta(df, "rolling_vol_short", ta.STDDEV, returns, roll_short),
                    self._ta(df, "rolling_vol_long", ta.STDDEV, returns, roll_long),
                ]).alias("volatility"),

                # --- Trend ---
                pl.struct([
                    ((df["ema_fast"] - df["ema_slow"]) / (df["ema_slow"].abs() + 1e-9)).alias("strength"),
                    (df["adx"] * ((df["ema_fast"] - df["ema_slow"]) / (df["ema_slow"].abs() + 1e-9))).alias("confidence"),

                    # Fast ↔ Mid cross
                    (
                        ( (df["ema_fast"] > df["ema_mid"]).cast(pl.Int8) - (df["ema_fast"].shift(1) > df["ema_mid"].shift(1)).cast(pl.Int8) )
                    ).alias("cross_fast_mid"),

                    # Mid ↔ Slow cross
                    (
                        ( (df["ema_mid"] > df["ema_slow"]).cast(pl.Int8) - (df["ema_mid"].shift(1) > df["ema_slow"].shift(1)).cast(pl.Int8) )
                    ).alias("cross_mid_slow"),

                    # Fast ↔ Slow cross
                    (
                        ( (df["ema_fast"] > df["ema_slow"]).cast(pl.Int8) - (df["ema_fast"].shift(1) > df["ema_slow"].shift(1)).cast(pl.Int8) )
                    ).alias("cross_fast_slow"),
                ]).alias("trend"),

                # --- Structure / Price Action ---
                pl.struct([
                    ((df["high"] - df["bb_upper"]) / (df["atr_pct"] + 1e-9)).alias("breakout_upper"),
                    ((df["low"] - df["bb_lower"]) / (df["atr_pct"] + 1e-9)).alias("breakout_lower"),
                    ((df["bb_width"] / self._ta(df, "bb_width_ma20", ta.SMA, df["bb_width"], 20)) - 1).alias("range_expansion"),
                ]).alias("structure"),

                # --- Volume / Market Activity ---
                pl.struct([
                    ((df["volume"] / self._ta(df, "volume_ma", ta.SMA, df["volume"], timeperiod=self.params.vol_sma)) - 1).alias("anomaly"),
                    ((df["obv"] - df["obv"].shift(5))/(self._ta(df, "obv_std", ta.STDDEV, df["obv"], timeperiod=20) + 1e-9)).alias("obv_momentum"),
                ]).alias("volume"),

                # --- Oscillators / Extremes ---
                pl.struct([
                    ((df["rsi_slow"] + df["stochK"]) / 2).alias("convergence"),
                    pl.when(df["rsi_slow"] > 70).then(1).when(df["rsi_slow"] < 30).then(-1).otherwise(0).alias("rsi_extreme"),
                ]).alias("oscillator"),
            ]).alias("derived")
        ])

    @staticmethod
    def _hv_proxy(df: pl.DataFrame, window: int = 50, annualize: int = 252) -> pl.Series:
        """
        Historical volatility proxy (like VIX) based on log returns.
        
        Args:
            df: input dataframe with price column
            col: column name to compute volatility on (default "close")
            window: rolling window length in bars
            annualize: annualization factor (default 252 trading days)
        
        Returns:
            pl.Series of historical volatility
        """
        price = df["close"].to_numpy()
        log_ret = np.diff(np.log(price), prepend=np.nan)
        hv = np.full(len(price), np.nan, dtype=np.float32)
        if len(price) >= window:
            for i in range(window-1, len(price)):
                hv[i] = np.nanstd(log_ret[i-window+1:i+1], ddof=1) * np.sqrt(annualize)
        return pl.Series(name=f"hv_{window}", values=hv)

    @staticmethod
    def swing_high_low(df: pl.DataFrame, window: int = 20, atr_mult: float = 1.0) -> pl.DataFrame:
        if df.is_empty():
            return df

        df = df.sort("date")
        ohlcv = Indicator._ohlcv(df)
        o, h, l, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        n = len(h)

        # --- Identify raw swing highs/lows ---
        roll_max = maximum_filter1d(h, size=2*window+1, mode="reflect")
        roll_min = minimum_filter1d(l, size=2*window+1, mode="reflect")
        swing = np.where(h == roll_max, 1, np.where(l == roll_min, -1, np.nan))

        # --- ATR-based filtering ---
        if "atr" in df.columns and atr_mult:
            atr = df["atr"].to_numpy()
            idx = np.flatnonzero(~np.isnan(swing))
            if idx.size > 1:
                prev_idx, curr_idx = idx[:-1], idx[1:]
                dist = np.where(swing[curr_idx] == 1, h[curr_idx] - l[prev_idx], h[prev_idx] - l[curr_idx])
                mask = dist < atr_mult * atr[curr_idx]
                swing[curr_idx[mask]] = np.nan

        # --- Remove consecutive swings in same direction keeping extremes ---
        idx = np.flatnonzero(~np.isnan(swing))
        if idx.size:
            clean_idx = [idx[0]]
            for i in idx[1:]:
                prev = clean_idx[-1]
                if swing[i] == swing[prev]:
                    if swing[i] == 1 and h[i] > h[prev]:
                        clean_idx[-1] = i
                    elif swing[i] == -1 and l[i] < l[prev]:
                        clean_idx[-1] = i
                else:
                    clean_idx.append(i)
            mask = np.full(n, False, dtype=bool)
            mask[clean_idx] = True
            swing = np.where(mask, swing, np.nan)

        # --- Levels ---
        level = np.where(swing == 1, h, np.where(swing == -1, l, np.nan))
        high = np.where(swing == 1, level, np.nan)
        low = np.where(swing == -1, level, np.nan)

        # --- Forward-fill last and active swings ---
        last = np.full(n, np.nan)
        active = np.full(n, np.nan)
        last_val = np.nan
        for i in range(n):
            if not np.isnan(swing[i]):
                last_val = level[i]
            last[i] = last_val
            active[i] = h[i] if last[i] == h[i] else (l[i] if last[i] == l[i] else np.nan)

        # --- Strength computation ---
        strength = np.full(n, np.nan, dtype=float)
        last_swing_val = np.nan
        for i in range(n):
            if not np.isnan(swing[i]):
                if not np.isnan(last_swing_val):
                    strength[i] = (h[i] - last_swing_val) if swing[i] == 1 else (last_swing_val - l[i])
                last_swing_val = level[i]
            else:
                strength[i] = np.nan
        if "atr" in df.columns:
            strength /= df["atr"].to_numpy() + 1e-9

        last_int = np.nan_to_num(last, nan=-1).astype("int8")

        # --- Build struct column properly using pl.struct and pl.Series ---
        return df.with_columns([
            pl.struct([
                pl.Series("high", high),
                pl.Series("low", low),
                pl.Series("last", last_int),
                pl.Series("active", active),
                pl.Series("strength", strength)
            ]).alias("swings")
        ])

    @staticmethod
    def _calc_fibs(df: pl.DataFrame, atr_mult: float = 0.0) -> pl.DataFrame:
        """
        Calculate Fibonacci retracement/extension levels from swings and attach as a
        single struct column 'fibs' containing:
          - f000, f236, f382, f500, f618, f786, f1000 (fib levels)
          - distance_raw: absolute distance to nearest fib
          - distance: ATR-normalized distance
          - ratio: close / nearest fib level
          - breakout: breakout direction (-1 = below, 0 = inside, 1 = above)
        """
        if df.is_empty(): return df
        active = df["swings"].struct.field("active").forward_fill()
        swing_dir = df["swings"].struct.field("last").forward_fill()

        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        fib_names = [f"f{int(f*1000):04d}" for f in fib_levels]

        # --- Build fib projections ---
        fib_exprs = []
        for f, name in zip(fib_levels, fib_names):
            if atr_mult > 0:
                if "atr" not in df.columns:
                    raise ValueError("ATR column required when atr_mult > 0")
                expr = (
                    pl.when(swing_dir == 1)
                      .then(active - f * atr_mult * df["atr"])
                      .when(swing_dir == -1)
                      .then(active + f * atr_mult * df["atr"])
                      .alias(name)
                )
            else:
                expr = (
                    pl.when(swing_dir == 1)
                      .then(active - (active - df["low"]) * f)
                      .when(swing_dir == -1)
                      .then(active + (df["high"] - active) * f)
                      .alias(name)
                )
            fib_exprs.append(expr)

        # Add fib columns
        df = df.with_columns(fib_exprs)
        fib_cols = [pl.col(name) for name in fib_names]

        # --- Derived features ---
        diffs = [(df["close"] - fib).abs() for fib in fib_cols]
        fib_distance_raw = pl.min_horizontal(*diffs).alias("distance_raw")
        nearest_fib = pl.min_horizontal(*fib_cols)  # <-- replace with argmin logic if label is needed
        fib_distance = (fib_distance_raw / (df["atr"] + 1e-9)).alias("distance")
        fib_ratio = (df["close"] / nearest_fib).alias("ratio")

        fib_breakout = (
            pl.when(df["close"] > pl.max_horizontal(*fib_cols)).then(1)
             .when(df["close"] < pl.min_horizontal(*fib_cols)).then(-1)
             .otherwise(0)
             .alias("breakout")
        )
        fibs_struct = pl.struct(fib_cols + [fib_distance_raw, fib_distance, fib_ratio, fib_breakout]).alias("fibs")
        return df.with_columns(fibs_struct).drop(fib_cols)

    @staticmethod
    def _daily(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty(): return df
        daily_stats = (
            df.group_by("date")
              .agg([
                  pl.max("high").alias("DH"),
                  pl.min("low").alias("DL"),
              ])
              .sort("date")
              .with_columns([
                  pl.col("DH").shift(1).alias("PDH"),
                  pl.col("DL").shift(1).alias("PDL"),
                  (pl.col("DH") - pl.col("DL")).alias("dayRange"),
                  (pl.col("DH").shift(1) - pl.col("DL").shift(1)).alias("prevDayRange"),
              ])
              .with_columns([
                  ((pl.col("DH") - pl.col("DL")) / (pl.col("DH").shift(1) - pl.col("DL").shift(1) + 1e-9))
                  .alias("rangeExpansionRatio")
              ])
              .with_columns([
                  pl.col(c).forward_fill().alias(c)
                  for c in ["DH","DL","PDH","PDL","dayRange","prevDayRange","rangeExpansionRatio"]
              ])
        )
        df = df.join(daily_stats, on="date", how="left")
        daily_struct_cols = [
            "DH","DL","PDH","PDL",
            "dayRange","prevDayRange","rangeExpansionRatio",
            "brokenHighPrev","brokenLowPrev",
            "newHighToday","newLowToday"
        ]
        df = df.with_columns([
            (pl.col("high") > pl.col("PDH")).cast(pl.Int8).fill_null(0).alias("brokenHighPrev"),
            (pl.col("low") < pl.col("PDL")).cast(pl.Int8).fill_null(0).alias("brokenLowPrev"),
            (pl.col("high") >= pl.col("DH")).cast(pl.Int8).alias("newHighToday"),
            (pl.col("low") <= pl.col("DL")).cast(pl.Int8).alias("newLowToday"),
        ])
        return df.with_columns(
            pl.struct(daily_struct_cols).alias("daily")
        ).drop(daily_struct_cols)

    def _htf_gap(self, df: pl.DataFrame, tf: str = None) -> pl.DataFrame:
        """
        Compute HTF gap metrics and pack into struct columns per HTF.
        Returns a struct named gaps.{tf} with:
          - prevHigh
          - prevLow
          - gap
          - gapPct
          - brokenHigh
          - brokenLow
        """
        if df.is_empty(): return df
        df = df.sort("date")
        if tf is None: tf = self.strategy.timeframe
        resampled = (
            df.group_by_dynamic(
                "date",
                every=tf,
                period=tf,
                include_boundaries=True
            ).agg([
                pl.col("high").max().alias("res_high"),
                pl.col("low").min().alias("res_low"),
            ]).sort("date")
        ).with_columns([
            pl.col("res_high").shift(1).fill_null(strategy="forward").alias("prevHigh"),
            pl.col("res_low").shift(1).fill_null(strategy="forward").alias("prevLow"),
        ])
        df = df.join_asof(
            resampled.select(["date", "prevHigh", "prevLow"]),
            on="date",
            strategy="backward"
        )
        gap_struct = pl.struct([
            pl.col("prevHigh"),
            pl.col("prevLow"),
            (pl.col("prevHigh") - pl.col("prevLow")).alias("gap"),
            ((pl.col("prevHigh") - pl.col("prevLow")) / (pl.col("prevLow") + 1e-9)).alias("gapPct"),
            (pl.col("high") > pl.col("prevHigh")).cast(pl.Int8).alias("brokenHigh"),
            (pl.col("low") < pl.col("prevLow")).cast(pl.Int8).alias("brokenLow"),
        ]).alias("gap")
        return df.with_columns(gap_struct).drop(["prevHigh", "prevLow"])

    @staticmethod
    def _fvg(df: pl.DataFrame, merge_consecutive: bool = False) -> pl.DataFrame:
        if df.is_empty(): return df
        ohlcv = Indicator._ohlcv(df)
        o, h, l, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
        n = len(df)

        fvg = np.full(n, np.nan, dtype=np.float32)
        bull_mask = (h[:-2] < l[2:]) & (c[1:-1] > o[1:-1])
        bear_mask = (l[:-2] > h[2:]) & (c[1:-1] < o[1:-1])
        fvg[1:-1][bull_mask] = 1
        fvg[1:-1][bear_mask] = -1

        high_arr = np.full(n, np.nan, dtype=np.float32)
        low_arr = np.full(n, np.nan, dtype=np.float32)

        bull_idx = np.flatnonzero(fvg == 1)
        bear_idx = np.flatnonzero(fvg == -1)

        high_arr[bull_idx] = l[bull_idx + 1]
        low_arr[bull_idx] = h[bull_idx - 1]

        high_arr[bear_idx] = l[bear_idx - 1]
        low_arr[bear_idx] = h[bear_idx + 1]

        if merge_consecutive:
            same = (fvg[1:] == fvg[:-1]) & ~np.isnan(fvg[1:])
            grp = (~same).cumsum()
            merged_high = np.full_like(high_arr, np.nan)
            merged_low = np.full_like(low_arr, np.nan)
            merged_fvg = np.full_like(fvg, np.nan)
            for g in np.unique(grp):
                mask = grp == g
                if np.any(~np.isnan(fvg[mask])):
                    last_idx = np.where(mask)[0][-1]
                    merged_fvg[last_idx] = fvg[mask][0]
                    merged_high[last_idx] = np.nanmax(high_arr[mask])
                    merged_low[last_idx] = np.nanmin(low_arr[mask])
            fvg, high_arr, low_arr = merged_fvg, merged_high, merged_low

        mitigated_idx = np.full(n, np.nan, dtype=np.float32)
        pct = np.zeros(n, dtype=np.float32)
        strength = np.full(n, np.nan, dtype=np.float32)

        rng = np.maximum(high_arr - low_arr, 1e-9)
        valid = ~np.isnan(rng)
        strength[valid] = rng[valid] / c[valid]

        def first_hit(idx_array, is_bull: bool):
            for idx in idx_array:
                top, bot = high_arr[idx], low_arr[idx]
                if np.isnan(top) or np.isnan(bot):
                    continue
                if is_bull:
                    sweep = np.where(l[idx + 2:] <= top)[0]
                    if sweep.size:
                        hit = idx + 2 + sweep[0]
                        mitigated_idx[idx] = hit
                        pct[idx] = np.clip((top - l[hit]) / rng[idx] * 100, 0, 100)
                else:
                    sweep = np.where(h[idx + 2:] >= bot)[0]
                    if sweep.size:
                        hit = idx + 2 + sweep[0]
                        mitigated_idx[idx] = hit
                        pct[idx] = np.clip((h[hit] - bot) / rng[idx] * 100, 0, 100)

        first_hit(bull_idx, True)
        first_hit(bear_idx, False)

        fvg_struct = pl.struct([
            pl.Series([None if np.isnan(x) else int(x) for x in fvg], dtype=pl.Int8).alias("fvg"),
            pl.Series([None if np.isnan(x) else int(x) for x in mitigated_idx], dtype=pl.Int32).alias("index"),
            pl.Series(strength).cast(pl.Float32).fill_nan(None).alias("strength"),
            pl.Series(high_arr).cast(pl.Float32).fill_nan(None).alias("high"),
            pl.Series(low_arr).cast(pl.Float32).fill_nan(None).alias("low"),
            pl.Series(pct).cast(pl.Float32).fill_nan(None).alias("pct"),
        ]).alias("fvg")

        return df.with_columns(fvg_struct)

    @staticmethod
    def previous_high_low(df: pl.DataFrame, timeframe: str = "15m") -> pl.DataFrame:
        if df.is_empty(): return df
        if timeframe not in {"15m", "1h", "4h", "1d", "1w"}:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        resampled = (
            df.sort("date").group_by_dynamic(
                    "date",
                    every=timeframe,
                    period=timeframe,
                    include_boundaries=True
                ).agg([
                    pl.col("open").first().alias("res_open"),
                    pl.col("high").max().alias("res_high"),
                    pl.col("low").min().alias("res_low"),
                    pl.col("close").last().alias("res_close"),
                    pl.col("volume").sum().alias("res_volume"),
                ]).sort("date")
        ).with_columns([
            pl.col("res_high").shift(1).alias("prevHigh"),
            pl.col("res_low").shift(1).alias("prevLow")
        ])
        intraday = df.join_asof(resampled.select(["date", "prevHigh", "prevLow"]), on="date", strategy="backward")
        return intraday.with_columns([
            (pl.col("high") > pl.col("prevHigh")).cast(pl.Int8).alias("brokenHigh"),
            (pl.col("low") < pl.col("prevLow")).cast(pl.Int8).alias("brokenLow")
        ]).select(["date", "prevHigh", "prevLow", "brokenHigh", "brokenLow"])

    @staticmethod
    def liquidity_zones(df: pl.DataFrame, swings: pl.DataFrame, atr_mult: float = 1.0) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame(
                {"date": [], "index": [], "zone": [], "level": [], "end": [], "pct": []}
            )

        atr_col = "atr"
        if not {"high", "low", atr_col}.issubset(df.columns):
            raise ValueError(f"df must contain 'high', 'low', '{atr_col}'")
        if not {"HighLow", "level"}.issubset(swings.columns):
            raise ValueError("swings must contain 'HighLow', 'level'")

        swings = swings.join(
            df.select("date").with_row_index("idx"), on="date", how="inner"
        )

        h = df["high"].to_numpy()
        l = df["low"].to_numpy()
        atr = df[atr_col].to_numpy() * atr_mult
        hl_type = swings["HighLow"].to_numpy()
        hl_level = swings["level"].to_numpy()
        idx = swings["idx"].to_numpy()
        n = len(df)

        zone_arr = np.full(n, np.nan, dtype=np.float32)
        level_arr = np.full(n, np.nan, dtype=np.float32)
        end_arr = np.full(n, np.nan, dtype=np.float32)
        swept_idx_arr = np.full(n, np.nan, dtype=np.float32)
        pct_arr = np.zeros(n, dtype=np.float32)

        valid_mask = np.isfinite(hl_type) & np.isfinite(hl_level)
        hl_type, hl_level, idx = hl_type[valid_mask], hl_level[valid_mask], idx[valid_mask]

        # process supply (+1) and demand (-1) separately
        for sign in (1.0, -1.0):
            mask = hl_type == sign
            if not np.any(mask):
                continue

            type_idx = idx[mask]
            levels = hl_level[mask]
            widths = atr[type_idx]

            # group by bin to avoid overlapping duplicate zones
            bins = np.floor(levels / widths)
            _, first_in_bin = np.unique(bins, return_index=True)

            for start, level, width in zip(
                type_idx[first_in_bin], levels[first_in_bin], widths[first_in_bin]
            ):
                if width <= 0 or np.isnan(width):
                    continue

                # find sweep point
                if sign == 1:  # supply (high side)
                    sweep = np.where(h[start + 1 :] >= level + width)[0]
                else:  # demand (low side)
                    sweep = np.where(l[start + 1 :] <= level - width)[0]

                if sweep.size:
                    end_idx = start + 1 + sweep[0]
                    swept_idx = end_idx
                else:
                    end_idx = n - 1
                    swept_idx = np.nan

                # progressive % penetration
                if sign == 1:
                    pct_slice = (h[start : end_idx + 1] - level) / width * 100
                else:
                    pct_slice = (level - l[start : end_idx + 1]) / width * 100

                pct_slice = np.nan_to_num(pct_slice, nan=0.0, posinf=100.0, neginf=0.0)
                pct_slice = np.clip(pct_slice, 0, 100)

                # fill arrays
                zone_arr[start : end_idx + 1] = sign
                level_arr[start : end_idx + 1] = level
                end_arr[start : end_idx + 1] = end_idx
                pct_arr[start : end_idx + 1] = pct_slice
                if not np.isnan(swept_idx):
                    swept_idx_arr[start : end_idx + 1] = swept_idx

        return pl.DataFrame(
            {
                "date": df["date"],
                "index": pl.Series(
                    [int(x) if np.isfinite(x) else None for x in swept_idx_arr],
                    dtype=pl.Int64,
                ),
                "zone": pl.Series(zone_arr).fill_nan(None),
                "level": pl.Series(level_arr).fill_nan(None),
                "end": pl.Series(end_arr).fill_nan(None),
                "pct": pl.Series(pct_arr).fill_nan(None),
            }
        )

    @staticmethod
    def order_blocks(df: pl.DataFrame, swings: pl.DataFrame, close_mitigation: bool = False) -> pl.DataFrame:
        n = df.height
        ob = np.full(n, np.nan, dtype=np.float32)
        top_arr = np.full(n, np.nan, dtype=np.float32)
        bottom_arr = np.full(n, np.nan, dtype=np.float32)
        ob_volume = np.full(n, np.nan, dtype=np.float32)
        mitigated_idx = np.full(n, np.nan, dtype=np.float32)
        pct = np.full(n, np.nan, dtype=np.float32)
        swing_origin = np.full(n, np.nan, dtype=np.float32)
        start_idx = np.full(n, np.nan, dtype=np.float32)
        strength = np.full(n, np.nan, dtype=np.float32)

        high, low, close, open_, volume = (
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            df["close"].to_numpy(),
            df["open"].to_numpy(),
            df["volume"].to_numpy(),
        )
        swingH, swingL, swing_idx = (
            swings["high"].to_numpy(),
            swings["low"].to_numpy(),
            swings["index"].to_numpy(),
        )

        def compute_pct(vols: np.ndarray, bullish: bool) -> float:
            """Balance between entry vs rest of OB volume."""
            if vols.size == 0 or vols.sum() == 0:
                return 100.0
            if bullish:
                low_vol, high_vol = vols[0], vols[1:].sum()
            else:
                high_vol, low_vol = vols[-1], vols[:-1].sum()
            big, small = max(low_vol, high_vol), min(low_vol, high_vol)
            return float(small / big * 100) if big > 0 else 100.0

        active: list[tuple[int, int]] = []  # (origin_idx, type)

        for i in range(n):
            # check active OBs for mitigation
            still_active = []
            for idx0, ob_type in active:
                if ob_type == 1:  # bullish
                    breached = (low[i] < bottom_arr[idx0]) if not close_mitigation else (min(open_[i], close[i]) < bottom_arr[idx0])
                else:  # bearish
                    breached = (high[i] > top_arr[idx0]) if not close_mitigation else (max(open_[i], close[i]) > top_arr[idx0])
                if breached:
                    mitigated_idx[idx0] = i - 1
                else:
                    # propagate OB properties
                    top_arr[i] = top_arr[idx0]
                    bottom_arr[i] = bottom_arr[idx0]
                    ob_volume[i] = ob_volume[idx0]
                    pct[i] = pct[idx0]
                    ob[i] = ob[idx0]
                    swing_origin[i] = swing_origin[idx0]
                    start_idx[i] = start_idx[idx0]
                    strength[i] = strength[idx0]
                    still_active.append((idx0, ob_type))
            active = still_active

            # --- New bullish OB ---
            if i > 0 and not np.isnan(swingH[i - 1]) and close[i] > swingH[i - 1]:
                idx0 = i - 1
                vols = volume[max(0, i - 2): i + 1]
                ob[idx0] = 1
                top_arr[idx0], bottom_arr[idx0] = swingH[idx0], low[idx0]
                ob_volume[idx0] = vols.sum()
                pct[idx0] = compute_pct(vols, bullish=True)
                swing_origin[idx0] = swing_idx[i - 1] if not np.isnan(swing_idx[i - 1]) else idx0
                start_idx[idx0] = idx0
                disp = max(0.0, close[i] - swingH[i - 1])
                strength[idx0] = (ob_volume[idx0] * disp) / max(pct[idx0], 1e-6)
                active.append((idx0, 1))

            # --- New bearish OB ---
            if i > 0 and not np.isnan(swingL[i - 1]) and close[i] < swingL[i - 1]:
                idx0 = i - 1
                vols = volume[max(0, i - 2): i + 1]
                ob[idx0] = -1
                top_arr[idx0], bottom_arr[idx0] = high[idx0], swingL[idx0]
                ob_volume[idx0] = vols.sum()
                pct[idx0] = compute_pct(vols, bullish=False)
                swing_origin[idx0] = swing_idx[i - 1] if not np.isnan(swing_idx[i - 1]) else idx0
                start_idx[idx0] = idx0
                disp = max(0.0, swingL[i - 1] - close[i])
                strength[idx0] = (ob_volume[idx0] * disp) / max(pct[idx0], 1e-6)
                active.append((idx0, -1))

        return pl.DataFrame({
            "date": df["date"],
            "index": pl.Series([int(x) if not np.isnan(x) else None for x in start_idx], dtype=pl.Int32),
            "ob": pl.Series([int(x) if not np.isnan(x) else None for x in ob], dtype=pl.Int8),
            "mitigated_idx": pl.Series([int(x) if not np.isnan(x) else None for x in mitigated_idx], dtype=pl.Int32),
            "high": pl.Series(top_arr, dtype=pl.Float32),
            "low": pl.Series(bottom_arr, dtype=pl.Float32),
            "volume": pl.Series(ob_volume, dtype=pl.Float32),
            "pct": pl.Series(pct, dtype=pl.Float32),
            "strength": pl.Series(strength, dtype=pl.Float32),
        })

    @staticmethod
    def bos_choch(df: pl.DataFrame, swings: pl.DataFrame) -> pl.DataFrame:
        n = len(df)
        last = swings["last"].to_numpy()
        active = swings["active"].to_numpy()
        idx = swings["index"].to_numpy()

        bos = np.full(n, np.nan, dtype=np.float32)
        choch = np.full(n, np.nan, dtype=np.float32)
        level_out = np.full(n, np.nan, dtype=np.float64)
        idx_out = np.full(n, np.nan, dtype=np.float32)
        pct_out = np.full(n, np.nan, dtype=np.float64)

        last_high, last_low = None, None
        last_bos, last_choch = None, None
        last_level, last_pct, last_idx = None, None, None

        for i in range(n):
            if not np.isnan(idx[i]) and not np.isnan(last[i]):
                if last[i] == 1:  # swing high
                    if last_low is not None:
                        if active[i] < last_low:  # BOS down
                            last_bos, last_choch = -1, None
                            last_level, last_idx = active[i], idx[i]
                            last_pct = (last_low - active[i]) / last_low * 100
                        elif active[i] > last_low:  # CHoCH up
                            last_bos, last_choch = None, 1
                            last_level, last_idx = active[i], idx[i]
                            last_pct = (active[i] - last_low) / last_low * 100
                    last_high = active[i]

                elif last[i] == -1:  # swing low
                    if last_high is not None:
                        if active[i] > last_high:  # BOS up
                            last_bos, last_choch = 1, None
                            last_level, last_idx = active[i], idx[i]
                            last_pct = (active[i] - last_high) / last_high * 100
                        elif active[i] < last_high:  # CHoCH down
                            last_bos, last_choch = None, -1
                            last_level, last_idx = active[i], idx[i]
                            last_pct = (last_high - active[i]) / last_high * 100
                    last_low = active[i]

            # propagate
            bos[i] = last_bos
            choch[i] = last_choch
            level_out[i] = last_level
            idx_out[i] = last_idx
            pct_out[i] = last_pct

        return pl.DataFrame({
            "date": df["date"],
            "index": pl.Series(idx_out).cast(pl.Int32, strict=False),
            "bos": pl.Series(bos).cast(pl.Int8, strict=False),
            "choch": pl.Series(choch).cast(pl.Int8, strict=False),
            "level": pl.Series(level_out),
            "pct": pl.Series(pct_out),
        })

    @staticmethod
    def zerolag(df: pl.DataFrame):
        zlema_length: int = 7
        zlema_atr: int = 2
        band_multiplier: float = 0.4
        range_length: int = 7
        range_factor: float = 0.75
        range_threshold: float = 1.6
        range_min_bars: int = 2
        entry_offset: float = 1.7

        n = df.height
        src = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        atr = df["atr"].to_numpy()
        ema_fast = df["ema_fast"].to_numpy()

        # --- Rolling high/low ---
        pad_len = range_length - 1
        high_pad = np.pad(high, (pad_len, 0), mode='edge')
        low_pad = np.pad(low, (pad_len, 0), mode='edge')
        high_range = sliding_window_view(high_pad, range_length).max(axis=1)
        low_range = sliding_window_view(low_pad, range_length).min(axis=1)

        mid_range = (high_range + low_range) / 2
        range_size = np.maximum(high_range - low_range, 1e-9)
        range_ratio = range_size / np.maximum(atr, 1e-9)
        range_dist = np.abs(src - mid_range) / range_size < 0.98
        is_range = (range_ratio < range_threshold) & range_dist
        is_range[:pad_len] = False  # first bars cannot be ranged

        # --- Consecutive range bars ---
        consecutive = np.zeros(n, dtype=int)
        mask = is_range.astype(int)
        last_zero = -1
        for i in range(n):
            if mask[i] == 0:
                consecutive[i] = 0
                last_zero = i
            else:
                consecutive[i] = i - last_zero
        ranging = is_range & (consecutive >= range_min_bars)

        # --- ZLEMA ---
        lag = (zlema_length - 1) // 2
        src_lagged = np.roll(src, lag)
        src_lagged[:lag] = src[:lag]
        zl_base = ema_fast + (ema_fast - src_lagged)
        kernel_len = max(1, zlema_length // 2)
        zlema = np.convolve(zl_base, np.ones(kernel_len) / kernel_len, mode='same')
        zlema[:kernel_len] = src[:kernel_len]

        # --- Bands ---
        avg_atr = np.convolve(atr, np.ones(zlema_atr) / zlema_atr, mode='same')
        zl_atr_ratio = np.clip(atr / np.maximum(avg_atr, 1e-9), 1, 2.5)
        volatility = avg_atr * band_multiplier
        zl_bandwidth = volatility * zl_atr_ratio
        zl_bandwidth[ranging] *= range_factor

        upper_band = zlema + zl_bandwidth
        lower_band = zlema - zl_bandwidth

        # --- Signals ---
        bull_signal = src > upper_band
        bear_signal = src < lower_band

        # --- Rolling one-hot signal ---
        state = np.zeros(n, dtype=int)  # 0 = none, 1 = bull, -1 = bear, 2 = range
        state[ranging] = 2
        state[bull_signal] = 1
        state[bear_signal] = -1

        for i in range(1, n):
            if state[i] == 0:
                state[i] = state[i - 1]

        ranging = state == 2
        bull_signal = state == 1
        bear_signal = state == -1

        # --- Trend & pre-range-trend ---
        trend = np.zeros(n, dtype=int)
        pre_range_trend = np.zeros(n, dtype=int)
        bars_since_trend = np.zeros(n, dtype=int)

        for i in range(1, n):
            if ranging[i]:
                trend[i] = 0
            elif bull_signal[i]:
                trend[i] = 1
            elif bear_signal[i]:
                trend[i] = -1
            else:
                trend[i] = trend[i - 1]

            bars_since_trend[i] = bars_since_trend[i - 1] + 1 if trend[i] == trend[i - 1] else 0
            pre_range_trend[i] = pre_range_trend[i - 1] if trend[i] == 0 else trend[i]

        # --- Adjust bands with entry_offset ---
        entry_offset_vec = np.where((trend != 0) & (pre_range_trend != trend) & (bars_since_trend <= 1), entry_offset, 1.0)
        upper_band = zlema + zl_bandwidth * entry_offset_vec
        lower_band = zlema - zl_bandwidth * entry_offset_vec

        return pl.DataFrame({
            "date": df["date"],
            "upper_band": upper_band,
            "lower_band": lower_band,
            "zlema": zlema,
            "trend": trend,
            "pre_range_trend": pre_range_trend,
            "is_range": ranging,
            "bull_signal": bull_signal,
            "bear_signal": bear_signal
        })

    @staticmethod
    def market(df: pl.DataFrame) -> pl.DataFrame:
        # df = self._calc_divergences(df, col="rsi")
        # df = self._calc_divergences(df, col="macd")
        # df = self._calc_value_area(df)

        shl = self.swing_high_low(df, 10)
        # lz = self.liquidity_zones(df, shl)
        # lz = lz.filter((lz["index"].is_not_null()))

        # phl = self.previous_high_low(df, '15m')
        # fibs = self.fibs(df, shl)
        # pdhl = self.daily_prev_high_low(df)
        # bc = self.bos_choch(df, shl)
        # bc = bc.filter((bc["bos"].is_not_null()) | (bc["choch"].is_not_null()))
        # fvg = self.fvg(df)
        # fvg = fvg.filter((fvg["index"].is_not_null()))
        # zlb = self.zerolag(df)
        # ob = self.order_blocks(df, shl)
        # ob = ob.filter((ob["swing_idx"].is_not_null()))

        # df = df.with_columns([
            # shl["high"].forward_fill(),
            # shl["low"].forward_fill(),
        # ])
        # print(shl)
        return shl

    @staticmethod
    def old_swing_high_low(df: pl.DataFrame, window: int = 5, atr_mult: float = 1.0) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame({
                "date": [], "index": [], "HighLow": [], "level": [],
                "high": [], "low": [], "last": [], "active": [], "strength": []
            })

        df = df.sort("date")
        h, l = df["high"].to_numpy(), df["low"].to_numpy()
        n = len(h)

        # --- Detect swing highs/lows ---
        roll_max = maximum_filter1d(h, size=2*window+1, mode="reflect")
        roll_min = minimum_filter1d(l, size=2*window+1, mode="reflect")
        swing = np.where(h == roll_max, 1, np.where(l == roll_min, -1, np.nan))

        # --- ATR-based filter for weak swings ---
        if "atr" in df.columns:
            atr = df["atr"].to_numpy()
            idx = np.flatnonzero(~np.isnan(swing))
            if idx.size > 1:
                prev_idx, curr_idx = idx[:-1], idx[1:]
                dist = np.where(
                    swing[curr_idx] == 1,  # high swing
                    h[curr_idx] - l[prev_idx],
                    h[prev_idx] - l[curr_idx]
                )
                mask = dist < atr_mult * atr[curr_idx]
                swing[curr_idx[mask]] = np.nan

        # --- Remove consecutive swings in same direction keeping most extreme ---
        idx = np.flatnonzero(~np.isnan(swing))
        if idx.size:
            clean_idx = [idx[0]]
            for i in idx[1:]:
                prev = clean_idx[-1]
                if swing[i] == swing[prev]:
                    if swing[i] == 1 and h[i] > h[prev]:
                        clean_idx[-1] = i
                    elif swing[i] == -1 and l[i] < l[prev]:
                        clean_idx[-1] = i
                else:
                    clean_idx.append(i)
            mask = np.full(n, False, dtype=bool)
            mask[clean_idx] = True
            swing = np.where(mask, swing, np.nan)

        # --- Base swing levels ---
        level = np.where(swing == 1, h, np.where(swing == -1, l, np.nan))
        index = np.where(~np.isnan(swing), np.arange(n), np.nan)

        df_out = pl.DataFrame({
            "date": df["date"],
            "index": pl.Series([None if np.isnan(x) else int(x) for x in index], dtype=pl.Int32),
            "HighLow": pl.Series([None if np.isnan(x) else int(x) for x in swing], dtype=pl.Int8),
            "level": pl.Series([None if np.isnan(x) else float(x) for x in level], dtype=pl.Float64),
        })

        # --- Forward-fill highs/lows & last swing ---
        df_out = df_out.with_columns([
            pl.when(pl.col("HighLow") == 1).then(pl.col("level")).alias("high"),
            pl.when(pl.col("HighLow") == -1).then(pl.col("level")).alias("low")
        ]).with_columns([
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("HighLow").forward_fill().alias("last")
        ]).with_columns(
            pl.when(pl.col("last") == 1)
              .then(pl.col("high"))
              .when(pl.col("last") == -1)
              .then(pl.col("low"))
              .alias("active")
        )

        # --- Swing strength (price movement from previous opposite swing) ---
        active_arr = df_out["active"].to_numpy()
        last_swing_val = np.nan
        strength = np.zeros(n, dtype=float)
        for i in range(n):
            if not np.isnan(swing[i]):
                if swing[i] == 1 and last_swing_val is not None:
                    strength[i] = h[i] - last_swing_val
                elif swing[i] == -1 and last_swing_val is not None:
                    strength[i] = last_swing_val - l[i]
                last_swing_val = level[i]
            else:
                strength[i] = np.nan
        if "atr" in df.columns:
            strength = strength / (atr + 1e-9)
        return df_out.with_columns(pl.Series("strength", strength))

    def tune(self, detector, n_trials: int = 30, path: str = None, direction: str = "maximize", scoring: str = "loglik"):
        """
        Tune IndicatorEngine params exclusively for regime detection.

        detector: RegimeDetector instance
        param_space: { "adx": (5, 20, "int"), ... }
        scoring: "loglik" | "silhouette" | callable(detector) -> float
        """
        from sklearn.metrics import silhouette_score

        def objective(trial):
            params = {}
            for name, (low, high, typ) in self.param_space.items():
                if typ == "int":
                    params[name] = trial.suggest_int(name, low, high)
                elif typ == "float":
                    params[name] = trial.suggest_float(name, low, high)

            self.params = SimpleNamespace(**{**vars(self.params), **params})
            detector = detector(self)

            try:
                detector.fit()
                feats = detector._scale_features()
                X = feats[detector._features_scaled_cols].to_numpy()

                if callable(scoring):
                    score = scoring(detector)
                elif scoring == "loglik":
                    score = detector.model.log_probability([X])
                elif scoring == "silhouette":
                    states = detector.model.predict([X])[0]
                    score = silhouette_score(X, states) if len(set(states)) > 1 else -1e6
                else:
                    raise ValueError(f"Unsupported scoring metric: {scoring}")

            except Exception:
                raise optuna.TrialPruned()

            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.params = SimpleNamespace(**{**vars(self.params), **best_params})

        if path:
            self.save(path)

        return best_params

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        fname = os.path.join(path, "indicator_params.json")
        with open(fname, "w") as f:
            json.dump(vars(self.params), f, indent=2)
        logger.info("IndicatorEngine params saved to %s", fname)

    def load(self, path: str):
        fname = os.path.join(path, "indicator_params.json")
        if not os.path.exists(fname):
            raise FileNotFoundError(f"No indicator_params.json found in {path}")

        with open(fname, "r") as f:
            loaded_params = json.load(f)

        merged = {**vars(self.params), **loaded_params}
        self.params = SimpleNamespace(**merged)

        extra_keys = set(loaded_params.keys()) - set(vars(self.params).keys())
        if extra_keys:
            logger.warning("Loaded params contain extra keys not in defaults: %s", extra_keys)

        logger.info("IndicatorEngine params loaded from %s", fname)
        return self.params
