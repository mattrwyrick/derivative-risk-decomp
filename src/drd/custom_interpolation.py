
"""
custom_interpolation.py

Curve smoothing & interpolation utilities using SciPy transforms
(FFT filtering, Savitzky–Golay, splines, Akima/PCHIP) and wavelet
denoising (PyWavelets). Also includes classic interpolants:
linear, log-linear, cubic spline, natural cubic spline, log-spline,
and step functions (forward/backward piecewise constant).

Great for noisy term-structure data (yield curves, DFs, vol smiles).

Dependencies:
    numpy
    scipy (scipy.interpolate, scipy.signal, scipy.fft)
    pywt  (PyWavelets)  # for wavelet denoising (optional)
    matplotlib (optional, demo only)

Usage:
    from custom_interpolation import CurveSmoother, smooth_and_interpolate

    x, y = ...  # strictly increasing x (tenors), y values
    sm = CurveSmoother(x, y)
    y_new = (sm.fft_lowpass(cutoff_frac=0.25)
                .pchip()
                .eval(new_x))

Notes:
- x must be strictly increasing; use `ensure_strictly_increasing` if needed.
- For positive quantities (discount factors, vols), log-linear/log-spline can be preferable.
- For monotone shapes (DFs), prefer PCHIP or step interpolants.
"""

from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Callable
import numpy as np

# Lazy imports for optional libs (so the file can import without SciPy at build time)
_sc_loaded = False
_pywt_loaded = False

def _load_scipy():
    global _sc_loaded, signal, interpolate, fft
    if not _sc_loaded:
        from scipy import signal, interpolate, fft
        _sc_loaded = True
    return signal, interpolate, fft

def _load_pywt():
    global _pywt_loaded, pywt
    if not _pywt_loaded:
        import pywt
        _pywt_loaded = True
    return pywt


# --------------------------- helpers ---------------------------

def ensure_strictly_increasing(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Force x to be strictly increasing by nudging ties forward by eps."""
    x = np.asarray(x, float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i-1]:
            x[i] = x[i-1] + eps
    return x

def _as_1d(a) -> np.ndarray:
    z = np.asarray(a, float).reshape(-1)
    if z.ndim != 1:
        raise ValueError("Input must be one-dimensional")
    return z


# --------------------------- core class ---------------------------

@dataclass
class CurveSmoother:
    x: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        self.x = ensure_strictly_increasing(_as_1d(self.x))
        self.y = _as_1d(self.y)
        if self.x.shape != self.y.shape:
            raise ValueError("x and y must have the same length")
        self._y_work = self.y.copy()
        self._interp_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    # ----- smoothing transforms -----

    def savgol(self, window: int = 5, polyorder: int = 2, deriv: int = 0) -> "CurveSmoother":
        """Savitzky–Golay filter (SciPy). window must be odd and >= polyorder+2."""
        signal, _, _ = _load_scipy()
        w = max(window, polyorder + 2)
        w = w + 1 if w % 2 == 0 else w
        self._y_work = signal.savgol_filter(self._y_work, window_length=w, polyorder=polyorder, deriv=deriv)
        return self

    def fft_lowpass(self, cutoff_frac: float = 0.25) -> "CurveSmoother":
        """
        FFT low-pass on irregular grid by interpolating to uniform grid, filtering, mapping back.
        cutoff_frac: 0..0.5 (ratio of Nyquist). 0.25 keeps lower 25% of frequencies.
        """
        _, _, fft = _load_scipy()
        n = len(self.x)
        u = np.linspace(self.x[0], self.x[-1], n)
        y_u = np.interp(u, self.x, self._y_work)
        Y = fft.rfft(y_u)
        freqs = fft.rfftfreq(n, d=(u[1]-u[0]))
        nyq = freqs.max() if freqs.size else 1.0
        mask = np.abs(freqs) <= cutoff_frac * nyq
        Y_f = Y * mask
        y_f = fft.irfft(Y_f, n=n)
        self._y_work = np.interp(self.x, u, y_f)
        return self

    def wavelet_denoise(self, wavelet: str = "db4", level: Optional[int] = None,
                        threshold_mode: Literal["universal","sureshrink","none"] = "universal",
                        soft: bool = True) -> "CurveSmoother":
        """Wavelet denoising via PyWavelets. Supports universal or SURE threshold selection."""
        pywt = _load_pywt()
        data = self._y_work
        coeffs = pywt.wavedec(data, wavelet=wavelet, level=level)
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size > 0 else 0.0

        def _thr(v, t):
            return np.sign(v) * np.maximum(np.abs(v)-t, 0.0) if soft else v * (np.abs(v) >= t)

        new_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            if threshold_mode == "none":
                new_coeffs.append(d)
            elif threshold_mode == "universal":
                t = sigma * np.sqrt(2.0 * np.log(len(data)))
                new_coeffs.append(_thr(d, t))
            elif threshold_mode == "sureshrink":
                v = np.sort(np.abs(d)); n = len(v); risks = []
                for k in range(n):
                    t = v[k]
                    indicator = (np.abs(d) <= t).astype(float)
                    risk = n - 2.0*np.sum(indicator) + np.sum(np.minimum(d*d, t*t))
                    risks.append(risk)
                t = v[int(np.argmin(risks))] if n>0 else 0.0
                new_coeffs.append(_thr(d, t))
            else:
                raise ValueError("Unknown threshold_mode")
        self._y_work = pywt.waverec(new_coeffs, wavelet=wavelet)[:len(self._y_work)]
        return self

    # ----- classic interpolants -----

    def linear(self) -> "CurveSmoother":
        _, itp, _ = _load_scipy()
        f = itp.interp1d(self.x, self._y_work, kind="linear", fill_value="extrapolate", assume_sorted=True)
        self._interp_fn = lambda xx: f(xx)
        return self

    def log_linear(self, eps: float = 1e-12) -> "CurveSmoother":
        """Linear interpolation in log(y); requires y>0; clamps via eps."""
        _, itp, _ = _load_scipy()
        y_pos = np.maximum(self._y_work, eps)
        f = itp.interp1d(self.x, np.log(y_pos), kind="linear", fill_value="extrapolate", assume_sorted=True)
        self._interp_fn = lambda xx: np.exp(f(xx))
        return self

    def spline(self, s: float = 0.0, k: int = 3) -> "CurveSmoother":
        """Univariate smoothing spline (k=1..5). s>0 enables smoothing."""
        _, itp, _ = _load_scipy()
        spl = itp.UnivariateSpline(self.x, self._y_work, k=k, s=max(0.0, s))
        self._interp_fn = lambda xx: spl(xx)
        return self

    def cubic_spline(self) -> "CurveSmoother":
        """C2 cubic spline (InterpolatedUnivariateSpline)."""
        _, itp, _ = _load_scipy()
        spl = itp.InterpolatedUnivariateSpline(self.x, self._y_work, k=3)
        self._interp_fn = lambda xx: spl(xx)
        return self

    def natural_cubic_spline(self) -> "CurveSmoother":
        """Natural cubic spline with zero second derivatives at the ends."""
        _, itp, _ = _load_scipy()
        cs = itp.CubicSpline(self.x, self._y_work, bc_type="natural", extrapolate=True)
        self._interp_fn = lambda xx: cs(xx)
        return self

    def log_spline(self, s: float = 0.0, k: int = 3, eps: float = 1e-12) -> "CurveSmoother":
        """Spline in log(y), then exponentiate; requires y>0."""
        _, itp, _ = _load_scipy()
        y_pos = np.maximum(self._y_work, eps)
        spl = itp.UnivariateSpline(self.x, np.log(y_pos), k=k, s=max(0.0, s))
        self._interp_fn = lambda xx: np.exp(spl(xx))
        return self

    def pchip(self) -> "CurveSmoother":
        """Shape-preserving monotone piecewise cubic (good for discount factors)."""
        _, itp, _ = _load_scipy()
        p = itp.PchipInterpolator(self.x, self._y_work, extrapolate=True)
        self._interp_fn = lambda xx: p(xx)
        return self

    def akima(self) -> "CurveSmoother":
        """Akima 1D interpolator (robust to outliers/spikes)."""
        _, itp, _ = _load_scipy()
        a = itp.Akima1DInterpolator(self.x, self._y_work)
        self._interp_fn = lambda xx: a(xx)
        return self

    def step_forward(self) -> "CurveSmoother":
        """
        Left-continuous piecewise-constant (a.k.a. forward fill / stair-step).
        For x in (x_i, x_{i+1}], returns y_i. Also y(x<=x0)=y0, y(x>xN)=yN.
        """
        x = self.x; y = self._y_work.copy()
        def f(xx: np.ndarray) -> np.ndarray:
            xx = _as_1d(xx)
            idx = np.searchsorted(x, xx, side="right") - 1
            idx = np.clip(idx, 0, len(x)-1)
            return y[idx]
        self._interp_fn = f
        return self

    def step_backward(self) -> "CurveSmoother":
        """
        Right-continuous piecewise-constant (a.k.a. backward fill).
        For x in [x_i, x_{i+1}), returns y_{i+1}. Also y(x<x0)=y0, y(x>=xN)=yN.
        """
        x = self.x; y = self._y_work.copy()
        def f(xx: np.ndarray) -> np.ndarray:
            xx = _as_1d(xx)
            idx = np.searchsorted(x, xx, side="left")
            idx = np.clip(idx, 0, len(x)-1)
            return y[idx]
        self._interp_fn = f
        return self

    # ----- post-processing -----

    def enforce_monotone(self, increasing: bool = True) -> "CurveSmoother":
        """Project y_work onto a monotone sequence (PAV)."""
        y = self._y_work.copy()
        if not increasing:
            y = -y
        n = len(y)
        vals = y.copy()
        wts = np.ones(n)
        i = 0
        while i < n-1:
            if vals[i] <= vals[i+1] + 1e-15:
                i += 1
                continue
            new_val = (wts[i]*vals[i] + wts[i+1]*vals[i+1])/(wts[i]+wts[i+1])
            vals[i] = new_val; wts[i] += wts[i+1]
            vals = np.delete(vals, i+1); wts = np.delete(wts, i+1)
            j = i
            while j>0 and vals[j-1] > vals[j] + 1e-15:
                new_val = (wts[j-1]*vals[j-1] + wts[j]*vals[j])/(wts[j-1]+wts[j])
                vals[j-1] = new_val; wts[j-1] += wts[j]
                vals = np.delete(vals, j); wts = np.delete(wts, j)
                j -= 1
            i = max(j, 0)
        yy = np.zeros_like(self._y_work)
        idx = 0
        for v, w in zip(vals, wts):
            w = int(w)
            yy[idx:idx+w] = v
            idx += w
        self._y_work = yy if increasing else -yy
        return self

    # ----- evaluate / export -----

    def eval(self, x_new: np.ndarray) -> np.ndarray:
        if self._interp_fn is None:
            self.linear()
        x_new = _as_1d(x_new)
        return _as_1d(self._interp_fn(x_new))

    def current(self) -> np.ndarray:
        """Return the current working y (after smoothing steps)."""
        return self._y_work.copy()

    def reset(self) -> "CurveSmoother":
        """Reset working series to original y."""
        self._y_work = self.y.copy()
        self._interp_fn = None
        return self


# --------------------------- functional API ---------------------------

def smooth_and_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray,
                           smooth: Literal["none","savgol","fft","wavelet"] = "savgol",
                           interp: Literal["pchip","akima","spline","linear","cubic",
                                           "natural_cubic","loglinear","logspline",
                                           "step_fwd","step_bwd"] = "pchip",
                           **kwargs) -> np.ndarray:
    """
    One-shot helper:
      - Apply a smoothing transform
      - Apply chosen interpolant
      - Evaluate on x_new

    kwargs forwarded to chosen methods, e.g.:
      cutoff_frac for 'fft', (window, polyorder) for 'savgol',
      s,k for 'spline'/'logspline', eps for log transforms.
    """
    sm = CurveSmoother(x, y)
    if smooth == "savgol":
        sm.savgol(window=kwargs.get("window", 7), polyorder=kwargs.get("polyorder", 2))
    elif smooth == "fft":
        sm.fft_lowpass(cutoff_frac=kwargs.get("cutoff_frac", 0.25))
    elif smooth == "wavelet":
        sm.wavelet_denoise(wavelet=kwargs.get("wavelet", "db4"),
                           level=kwargs.get("level", None),
                           threshold_mode=kwargs.get("threshold_mode", "universal"),
                           soft=kwargs.get("soft", True))
    elif smooth == "none":
        pass
    else:
        raise ValueError("Unknown smoothing mode")

    if interp == "pchip":
        sm.pchip()
    elif interp == "akima":
        sm.akima()
    elif interp == "spline":
        sm.spline(s=kwargs.get("s", 0.0), k=kwargs.get("k", 3))
    elif interp == "linear":
        sm.linear()
    elif interp == "cubic":
        sm.cubic_spline()
    elif interp == "natural_cubic":
        sm.natural_cubic_spline()
    elif interp == "loglinear":
        sm.log_linear(eps=kwargs.get("eps", 1e-12))
    elif interp == "logspline":
        sm.log_spline(s=kwargs.get("s", 0.0), k=kwargs.get("k", 3), eps=kwargs.get("eps", 1e-12))
    elif interp == "step_fwd":
        sm.step_forward()
    elif interp == "step_bwd":
        sm.step_backward()
    else:
        raise ValueError("Unknown interpolation mode")

    return sm.eval(x_new)


# --------------------------- demo ---------------------------

if __name__ == "__main__":
    # Demo: noisy yield curve → denoise + diverse interpolants
    import sys
    print("custom_interpolation demo")
    try:
        import matplotlib.pyplot as plt
        HAVE_PLT = True
    except Exception:
        HAVE_PLT = False

    rng = np.random.default_rng(42)
    x = np.array([0.25,0.5,1,2,3,5,7,10,15,20,30], float)
    true = 0.03 + 0.015*(1 - np.exp(-x/4.0))
    noise = rng.normal(0, 0.0008, size=x.size)
    y = true + noise
    x_new = np.linspace(x[0], x[-1], 400)

    from math import exp
    y_fft_pchip = smooth_and_interpolate(x, y, x_new, smooth="fft", interp="pchip", cutoff_frac=0.2)
    y_wav_akima = smooth_and_interpolate(x, y, x_new, smooth="wavelet", interp="akima", wavelet="db4")
    y_sg_spline = smooth_and_interpolate(x, y, x_new, smooth="savgol", interp="spline", window=7, polyorder=2, s=1e-6)
    y_loglin    = smooth_and_interpolate(x, np.maximum(y,1e-6), x_new, smooth="none", interp="loglinear")
    y_logspline = smooth_and_interpolate(x, np.maximum(y,1e-6), x_new, smooth="none", interp="logspline", s=0.0)

    if HAVE_PLT:
        plt.figure(figsize=(10,6))
        plt.plot(x, y, "o", label="noisy input")
        plt.plot(x_new, y_fft_pchip, label="FFT→PCHIP")
        plt.plot(x_new, y_wav_akima, label="Wavelet→Akima")
        plt.plot(x_new, y_sg_spline, label="SavGol→Spline(s=1e-6)")
        plt.plot(x_new, y_loglin, label="Log-Linear")
        plt.plot(x_new, y_logspline, label="Log-Spline")
        plt.plot(x_new, true, "--", label="true (unknown)")
        plt.legend()
        plt.title("Custom Interpolation & Smoothing")
        plt.tight_layout()
        plt.show()
    else:
        print("Matplotlib not available; computed demo series.")
