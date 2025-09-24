
"""
sofr_pricing_quantlib_rateslib.py

Recreates pricing + Greeks + carry/roll-down for:
  • SOFR 3M futures
  • Futures options (Black-76)
  • OIS (SOFR) swaps
  • Caps/Floors (sum of Black caplets)

Primary engine: QuantLib (curves, Black-76, discounting).
Optional: rateslib (if installed) for swaps; code is guarded to avoid hard dependency.

Install:
    pip install QuantLib-Python pandas numpy
    # optional:
    pip install rateslib

Run:
    python sofr_pricing_quantlib_rateslib.py

Inputs (optional): same CSVs as before in the working folder:
    - ois_par_quotes.csv
    - sofr_futures.csv
    - sofr_futures_options.csv
    - swap_trades.csv
    - cap_surface_quotes.csv
If not present, the script runs a self-contained demo (flat 4% curve).

"""

import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

try:
    import QuantLib as ql
    QL_AVAILABLE = True
except Exception as e:
    QL_AVAILABLE = False
    _QL_ERR = repr(e)

try:
    import rateslib as rl
    RL_AVAILABLE = True
except Exception as e:
    RL_AVAILABLE = False
    _RL_ERR = repr(e)


# ------------------------ Utilities ------------------------

def yf(d0: dt.date, d1: dt.date, basis: str = "ACT/365F") -> float:
    b = basis.upper()
    if b == "ACT/365F": return (d1 - d0).days / 365.0
    if b == "ACT/360":  return (d1 - d0).days / 360.0
    if b in ("30/360", "30U/360"):
        d0d = 30 if d0.day == 31 else d0.day
        d1d = 30 if (d1.day == 31 and d0d in (30,31)) else d1.day
        return (360*(d1.year-d0.year) + 30*(d1.month-d0.month) + (d1d-d0d))/360.0
    raise ValueError("Unsupported basis")

def third_wednesday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(2 - d.weekday()) % 7)  # Wed=2

def next_imm(d: dt.date) -> dt.date:
    for m in (3,6,9,12):
        cand = third_wednesday(d.year, m)
        if cand > d: return cand
    return third_wednesday(d.year+1, 3)

def sofr_futures_window(imm_start: dt.date) -> Tuple[dt.date, dt.date]:
    return imm_start, next_imm(imm_start)


# ------------------------ Curve: QuantLib build ------------------------

@dataclass
class QLCurve:
    ts: "ql.YieldTermStructure"

    def df(self, t: float) -> float:
        return float(self.ts.discount(t))

    def fwd_simple(self, t0: float, t1: float) -> float:
        P0 = self.df(t0); P1 = self.df(t1); tau = max(1e-12, t1 - t0)
        return (P0/P1 - 1.0)/tau

def build_curve_from_ois_par(valuation_date: dt.date,
                             maturities_years: List[float],
                             par_rates: List[float],
                             fixed_freq_per_year: int = 2) -> QLCurve:
    """
    Lightweight par-OIS bootstrap to discount nodes, then wrap in QuantLib DiscountCurve.
    Keeps dependencies simple but lets QuantLib handle discounting/forwards robustly.
    """
    assert QL_AVAILABLE, "QuantLib-Python is required. Install QuantLib-Python."
    nodes_t, nodes_df = [], []
    for T, K in zip(maturities_years, par_rates):
        n = int(round(T * fixed_freq_per_year))
        alpha = [1.0/fixed_freq_per_year]*max(1, n)
        times = [i/fixed_freq_per_year for i in range(1, n+1)]
        if len(nodes_t) == 0:
            S = sum(a * math.exp(-K*ti) for a, ti in zip(alpha[:-1], times[:-1])) if n > 1 else 0.0
            Pn = (1.0 - K*S) / (1.0 + K*alpha[-1])
        else:
            def P_of(t):
                if t <= nodes_t[0]: return nodes_df[0]
                if t >= nodes_t[-1]: return nodes_df[-1]
                lnP = np.interp(t, nodes_t, np.log(np.maximum(nodes_df, 1e-300)))
                return float(math.exp(lnP))
            S = sum(a * P_of(t) for a, t in zip(alpha[:-1], times[:-1])) if n > 1 else 0.0
            Pn = (1.0 - K*S) / (1.0 + K*alpha[-1])
        nodes_t.append(T); nodes_df.append(Pn)

    ql.Settings.instance().evaluationDate = ql.Date(valuation_date.day, valuation_date.month, valuation_date.year)
    ts = ql.DiscountCurve([ql.Time(t) for t in nodes_t], nodes_df, ql.Actual365Fixed(), ql.Compounded, ql.Annual)
    ts.enableExtrapolation()
    return QLCurve(ts)


# ------------------------ Products: Pricing & Greeks ------------------------

# Futures
def price_sofr_future(curve: QLCurve, T0: float, T1: float) -> Dict[str, float]:
    fwd = curve.fwd_simple(T0, T1)
    return {"forward_decimal": fwd, "price": 100.0 - 100.0*fwd}

# Futures options (Black-76 via QuantLib blackFormula)
@dataclass
class Black76:
    price: float; delta: float; gamma: float; vega: float; d1: float; d2: float

def black76_option_on_fut(F: float, K: float, T: float, sigma: float, is_call: bool, df_settle: float = 1.0) -> Black76:
    assert QL_AVAILABLE, "QuantLib-Python required for Black-76; install QuantLib-Python"
    opt_type = ql.Option.Call if is_call else ql.Option.Put
    value = ql.blackFormula(opt_type, K, F, sigma*math.sqrt(T), df_settle)
    vT = sigma*math.sqrt(T)
    d1 = (math.log(F/K)+0.5*sigma*sigma*T)/vT; d2 = d1 - vT
    N = lambda x: 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
    n = lambda x: (1.0/math.sqrt(2.0*math.pi))*math.exp(-0.5*x*x)
    delta = df_settle*( N(d1) if is_call else -N(-d1) )
    gamma = df_settle*n(d1)/(F*vT)
    vega  = df_settle*F*n(d1)*math.sqrt(T)
    return Black76(value, delta, gamma, vega, d1, d2)

# Swaps (par, PV, DV01 via QuantLib discounting)
@dataclass
class SwapPV:
    par: float; pv: float; annuity: float; dv01: float

def price_ois_swap(curve: QLCurve, pay_times: List[float], accruals: List[float], K: Optional[float] = None) -> SwapPV:
    discounts = np.array([curve.df(t) for t in pay_times])
    annuity = float(np.sum(np.array(accruals)*discounts))
    Pn = curve.df(pay_times[-1])
    par = (1.0 - Pn)/annuity
    if K is None:
        pv = 0.0; K = par
    else:
        pv = K*annuity - (1.0 - Pn)
    dv01 = annuity*1e-4
    return SwapPV(par, pv, annuity, dv01)

# Caps/Floors (caplet aggregation; Black on forward rate)
@dataclass
class CapletGreek:
    price: float; delta_fwd: float; gamma_fwd: float; vega: float

def price_caplet_black(df_start: float, df_end: float, T_start: float, T_end: float, K: float, sigma: float, is_caplet: bool) -> CapletGreek:
    tau = max(1e-12, T_end - T_start)
    L = (df_start/df_end - 1.0)/tau
    Texp = T_start
    if Texp <= 0 or sigma <= 0 or K <= 0 or L <= 0:
        return CapletGreek(0,0,0,0)
    vT = sigma*math.sqrt(Texp)
    d1 = (math.log(L/K)+0.5*sigma*sigma*Texp)/vT; d2 = d1 - vT
    N = lambda x: 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
    n = lambda x: (1.0/math.sqrt(2.0*math.pi))*math.exp(-0.5*x*x)
    df_pay = df_end
    if is_caplet:
        price = df_pay * tau * (L*N(d1) - K*N(d2))
        delta = df_pay * tau * N(d1)
    else:
        price = df_pay * tau * (K*N(-d2) - L*N(-d1))
        delta = -df_pay * tau * N(-d1)
    gamma = df_pay * tau * n(d1)/(L*vT)
    vega  = df_pay * tau * L*n(d1)*math.sqrt(Texp)
    return CapletGreek(price, delta, gamma, vega)


# ------------------------ Carry / Roll-down ------------------------

def carry_pnl(pv_func, dt_years: float = 1.0/252) -> float:
    return pv_func(dt_years) - pv_func(0.0)

def swap_pv_with_horizon(curve: QLCurve, K: float, pay_times: List[float], accruals: List[float]):
    base = list(zip(pay_times, accruals))
    def pv_at(h: float) -> float:
        t = [max(0.0, ti-h) for ti,_ in base if ti-h>1e-8]
        a = [aj for (ti,aj) in zip(pay_times, accruals) if ti-h>1e-8]
        if not t: return 0.0
        discounts = np.array([curve.df(x) for x in t])
        annuity = float(np.sum(np.array(a)*discounts))
        Pn = curve.df(t[-1])
        return K*annuity - (1.0 - Pn)
    return pv_at

def roll_down_forward(curve: QLCurve, T1: float, T2: float) -> float:
    return curve.fwd_simple(T1, T2)


# ------------------------ (Optional) rateslib equivalents ------------------------
# These are guarded—only used if rateslib is installed. They demonstrate how to
# wire a swap and compute PV/DV01 with rateslib's abstractions.
def rateslib_swap_dv01_example(val_date: dt.date, times: List[float], accruals: List[float], K: float, df_nodes_t: List[float], df_nodes_P: List[float]) -> Optional[Dict[str, float]]:
    if not RL_AVAILABLE:
        return None
    # Build a rateslib Curve from DF nodes using log-linear interpolation in time
    # Map node times to calendar dates (ACT/365F) for illustration
    nodes = {}
    for t, P in zip(df_nodes_t, df_nodes_P):
        d = val_date + dt.timedelta(days=int(round(t*365)))
        nodes[d] = P
    curve = rl.Curve(nodes=nodes, interpolation="log_linear", convention="act365f")
    # Build an IRS: fixed leg with those payment dates; floating leg as RFR (SOFR) is abstracted here
    # rateslib typically builds from start/end dates & frequency; here we approximate using tenor.
    start = val_date; end = val_date + dt.timedelta(days=int(round(times[-1]*365)))
    irs = rl.IRS(effective=start, termination=end, fixed_rate=K, notional=1.0, float_spread=0.0, convention="30/360", calendar=None, frequency="S")
    # Value with discount curve for both legs (simplification)
    res = irs.npv(curve=curve)
    # Parallel DV01 via bump (1bp)
    bumped = rl.Curve(nodes={d: P*math.exp(-0.0001* yf(val_date, d, "ACT/365F")) for d,P in nodes.items()}, interpolation="log_linear", convention="act365f")
    res_b = irs.npv(curve=bumped)
    return {"pv": float(res), "dv01_parallel": float(res_b - res)}

# ------------------------ CSV-driven or demo run ------------------------

def main():
    if not QL_AVAILABLE:
        raise RuntimeError("QuantLib-Python is required. Install QuantLib-Python. Error: %s" % _QL_ERR)

    # Try to load CSV OIS par quotes first
    try:
        ois = pd.read_csv("ois_par_quotes.csv", parse_dates=["valuation_date"])
        val_date = ois["valuation_date"].dt.date.iloc[0]
        curve = build_curve_from_ois_par(val_date,
                                         maturities_years=ois["maturity_years"].tolist(),
                                         par_rates=ois["par_rate_decimal"].tolist(),
                                         fixed_freq_per_year=int(ois["fixed_freq_per_year"].iloc[0]))
        df_nodes_t = ois["maturity_years"].tolist()
        df_nodes_P = [curve.df(t) for t in df_nodes_t]
    except Exception:
        # Flat 4% demo
        val_date = dt.date.today()
        ql.Settings.instance().evaluationDate = ql.Date(val_date.day, val_date.month, val_date.year)
        flat = ql.FlatForward(ql.Settings.instance().evaluationDate, 0.04, ql.Actual365Fixed())
        flat.enableExtrapolation()
        curve = QLCurve(flat)
        df_nodes_t = [0.5,1,2,3,5,7,10]
        df_nodes_P = [curve.df(t) for t in df_nodes_t]

    # ---- Futures ----
    imm0 = next_imm(val_date)
    start, end = sofr_futures_window(imm0)
    T0 = yf(val_date, start, "ACT/365F")
    T1 = yf(val_date, end, "ACT/365F")
    fut = price_sofr_future(curve, T0, T1)

    # ---- Futures Option (ATM) ----
    F = fut["forward_decimal"]; Texp = T0; K = F; sigma = 0.01; df_settle = curve.df(Texp)
    call = black76_option_on_fut(F, K, Texp, sigma, True, df_settle)
    put  = black76_option_on_fut(F, K, Texp, sigma, False, df_settle)

    # ---- Swap (2Y semiannual) ----
    pay_times = [0.5,1.0,1.5,2.0]; accr = [0.5]*4
    swp = price_ois_swap(curve, pay_times, accr)
    pv_fn = swap_pv_with_horizon(curve, swp.par, pay_times, accr)
    carry_1bd = carry_pnl(pv_fn, 1.0/252)

    # ---- Cap (2Y quarterly) ----
    q_times = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]; q_accr = [0.25]*8; K_cap = 0.05; vols = [0.01]*len(q_times)
    cap_price = 0.0; cap_vega = 0.0; prev = 0.0
    for t, vol in zip(q_times, vols):
        df0, df1 = curve.df(prev), curve.df(t)
        gres = price_caplet_black(df0, df1, prev, t, K_cap, vol, True)
        cap_price += gres.price; cap_vega += gres.vega; prev = t

    # ---- Optional: rateslib quick example (if installed) ----
    rl_info = None
    if RL_AVAILABLE:
        try:
            rl_info = rateslib_swap_dv01_example(val_date, pay_times, accr, swp.par, df_nodes_t, df_nodes_P)
        except Exception as e:
            rl_info = {"error": repr(e), "hint": "rateslib API may vary by version; adapt node mapping if needed."}

    # Print a compact summary
    print("=== QuantLib Results ===")
    print(f"Futures F={F:.6f}, Price={fut['price']:.4f}")
    print(f"Fut Opt (ATM) Call: price={call.price:.8f}  Δ={call.delta:.6f}  Γ={call.gamma:.6f}  Vega={call.vega:.6f}")
    print(f"Fut Opt (ATM) Put : price={put.price:.8f}  Δ={put.delta:.6f}  Γ={put.gamma:.6f}  Vega={put.vega:.6f}")
    print(f"Swap 2Y: par={swp.par:.6%}, annuity={swp.annuity:.6f}, DV01={swp.dv01:.8f}, carry(1bd)={carry_1bd:.8f}")
    print(f"Cap 2Y (K=5%): price={cap_price:.8f}, vega_sum={cap_vega:.6f}")
    if rl_info is not None:
        print("\n=== rateslib (optional) ===")
        print(rl_info)

if __name__ == "__main__":
    main()
