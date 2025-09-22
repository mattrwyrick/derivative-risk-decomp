
"""
sofr_pricing_lib.py  (enhanced)

Adds:
- US holiday generator (approx NY Fed schedule; observed rules)
- IMM → CME futures code mapping (SR3 month codes) and pack/bundle labels
- Key-rate PV01 report as a pandas.DataFrame
- Retains all pricing functionality from the prior version

Note: The holiday generator is a pragmatic approximation for trading calendars.
Always validate against your firm’s calendar service for production.
"""

import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np

try:
    import pandas as pd
except ImportError:  # optional
    pd = None

# ===================== Math =====================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0.0 else default

# ===================== Day-counts =====================

class DayCount:
    @staticmethod
    def act_360(d0: dt.date, d1: dt.date) -> float:
        return (d1 - d0).days / 360.0
    @staticmethod
    def act_365f(d0: dt.date, d1: dt.date) -> float:
        return (d1 - d0).days / 365.0
    @staticmethod
    def d30_360_us(d0: dt.date, d1: dt.date) -> float:
        d0d = 30 if d0.day == 31 else d0.day
        d1d = 30 if (d1.day == 31 and d0d in (30,31)) else d1.day
        return (360*(d1.year-d0.year) + 30*(d1.month-d0.month) + (d1.d - d0d)) / 360.0

def yf(d0: dt.date, d1: dt.date, basis: str = "ACT/360") -> float:
    b = basis.upper()
    if b == "ACT/360":
        return DayCount.act_360(d0, d1)
    if b == "ACT/365F":
        return DayCount.act_365f(d0, d1)
    if b in ("30/360","30E/360","30U/360"):
        # Using US convention here
        d0d = 30 if d0.day == 31 else d0.day
        d1d = 30 if (d1.day == 31 and d0d in (30,31)) else d1.day
        return (360*(d1.year-d0.year) + 30*(d1.month-d0.month) + (d1d - d0d)) / 360.0
    raise ValueError("Unsupported basis: " + basis)

# ===================== US Holiday (approx) =====================

def nth_weekday_of_month(year:int, month:int, weekday:int, n:int)->dt.date:
    """weekday: Mon=0..Sun=6; n>=1"""
    d = dt.date(year, month, 1)
    add = (weekday - d.weekday()) % 7
    d = d + dt.timedelta(days=add)  # first such weekday
    return d + dt.timedelta(days=7*(n-1))

def last_weekday_of_month(year:int, month:int, weekday:int)->dt.date:
    d = dt.date(year, month+1, 1) - dt.timedelta(days=1) if month<12 else dt.date(year, 12, 31)
    while d.weekday()!=weekday:
        d -= dt.timedelta(days=1)
    return d

def observed(d: dt.date)->dt.date:
    # Saturday → Friday, Sunday → Monday
    if d.weekday()==5:
        return d - dt.timedelta(days=1)
    if d.weekday()==6:
        return d + dt.timedelta(days=1)
    return d

def us_fed_holidays(year:int)->set:
    """Approximate NY Fed banking holidays for a given year."""
    H = set()
    # New Year's Day
    H.add(observed(dt.date(year,1,1)))
    # MLK (3rd Mon Jan)
    H.add(nth_weekday_of_month(year,1,0,3))
    # Presidents' Day (3rd Mon Feb)
    H.add(nth_weekday_of_month(year,2,0,3))
    # Memorial Day (last Mon May)
    H.add(last_weekday_of_month(year,5,0))
    # Juneteenth (June 19)
    H.add(observed(dt.date(year,6,19)))
    # Independence Day
    H.add(observed(dt.date(year,7,4)))
    # Labor Day (1st Mon Sep)
    H.add(nth_weekday_of_month(year,9,0,1))
    # Columbus/Indigenous Day (2nd Mon Oct)
    H.add(nth_weekday_of_month(year,10,0,2))
    # Veterans Day (Nov 11)
    H.add(observed(dt.date(year,11,11)))
    # Thanksgiving (4th Thu Nov)
    H.add(nth_weekday_of_month(year,11,3,4))
    # Christmas
    H.add(observed(dt.date(year,12,25)))
    return H

class BusinessCalendar:
    def __init__(self, holidays: Optional[set] = None):
        self.holidays = holidays or set()
    @staticmethod
    def is_weekend(d: dt.date) -> bool:
        return d.weekday() >= 5
    def is_holiday(self, d: dt.date) -> bool:
        return d in self.holidays
    def is_business_day(self, d: dt.date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)
    def adjust(self, d: dt.date, convention: str = "following") -> dt.date:
        if self.is_business_day(d):
            return d
        if convention == "following":
            dd = d
            while not self.is_business_day(dd):
                dd += dt.timedelta(days=1)
            return dd
        elif convention == "preceding":
            dd = d
            while not self.is_business_day(dd):
                dd -= dt.timedelta(days=1)
            return dd
        else:
            raise ValueError("Unsupported convention")
    def advance_bd(self, d: dt.date, n: int) -> dt.date:
        dd, step, left = d, (1 if n>=0 else -1), abs(n)
        while left > 0:
            dd += dt.timedelta(days=step)
            if self.is_business_day(dd):
                left -= 1
        return dd

def nyc_calendar_for_years(start_year:int, end_year:int)->BusinessCalendar:
    hol = set()
    for y in range(start_year, end_year+1):
        hol |= us_fed_holidays(y)
    return BusinessCalendar(holidays=hol)

NYC = nyc_calendar_for_years(dt.date.today().year-1, dt.date.today().year+5)

# ===================== IMM & CME Codes =====================

def third_wednesday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    offset = (2 - d.weekday()) % 7  # 2=Wed
    return d + dt.timedelta(days=offset)

def imm_quarterly_dates(start_year: int, end_year: int) -> List[dt.date]:
    imms = []
    for y in range(start_year, end_year+1):
        for m in (3,6,9,12):
            imms.append(third_wednesday(y, m))
    return sorted(imms)

def next_imm(d: dt.date) -> dt.date:
    for x in imm_quarterly_dates(d.year, d.year+1):
        if x > d:
            return x
    return imm_quarterly_dates(d.year+1, d.year+1)[0]

def sofr_futures_3m_window(imm_date: dt.date) -> Tuple[dt.date, dt.date]:
    return imm_date, next_imm(imm_date)

_MONTH_CODE = {3:"H", 6:"M", 9:"U", 12:"Z"}

def cme_sr3_code_from_imm(imm_date: dt.date) -> str:
    """Return CME 3M SOFR futures code like SR3H26 from IMM date."""
    return f"SR3{_MONTH_CODE[imm_date.month]}{str(imm_date.year)[-2:]}"

def pack_name_for_year_offset(offset:int)->str:
    """0=White (yr1), 1=Red (yr2), 2=Green (yr3), 3=Blue (yr4), 4=Gold (yr5), 5=Purple (yr6)."""
    names = ["White","Red","Green","Blue","Gold","Purple"]
    return names[offset] if 0 <= offset < len(names) else f"Pack{offset+1}"

def imm_packs(start_imm: dt.date, years: int = 1) -> List[List[dt.date]]:
    seq = []
    d = start_imm if start_imm == third_wednesday(start_imm.year, start_imm.month) else next_imm(start_imm)
    for _ in range(4*years):
        seq.append(d)
        d = next_imm(d)
    return [seq[i:i+4] for i in range(0, len(seq), 4)]

# ===================== Curves =====================

@dataclass
class DiscountCurve:
    df: Callable[[float], float]
    @staticmethod
    def from_flat_rate(r: float, comp: str = "cont"):
        if comp == "cont":
            return DiscountCurve(lambda t: math.exp(-r*t))
        elif comp == "simple":
            return DiscountCurve(lambda t: 1.0/(1.0 + r*t) if t!=0 else 1.0)
        else:
            raise ValueError("comp must be 'cont' or 'simple'")
    def P(self, t: float) -> float:
        return self.df(t)
    def fwd_simple(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 must be > t1")
        P1, P2 = self.P(t1), self.P(t2)
        tau = t2 - t1
        return (P1/P2 - 1.0)/tau
    def zero_rate_cont(self, t: float) -> float:
        if t <= 0:
            return 0.0
        return -math.log(max(self.P(t), 1e-300))/t

class PiecewiseFlatZeroCurve:
    def __init__(self, tenors: List[float], zeros: List[float]):
        assert len(tenors)==len(zeros)>=1
        self.tenors = np.array(tenors, float)
        self.zeros  = np.array(zeros, float)
        idx = np.argsort(self.tenors)
        self.tenors = self.tenors[idx]; self.zeros = self.zeros[idx]
    def zero(self, t: float) -> float:
        if t <= self.tenors[0]:
            return float(self.zeros[0])
        for i in range(1, len(self.tenors)):
            if t <= self.tenors[i]:
                return float(self.zeros[i])
        return float(self.zeros[-1])
    def P(self, t: float) -> float:
        z = self.zero(t)
        return math.exp(-z*t)
    def to_discount_curve(self) -> DiscountCurve:
        return DiscountCurve(lambda t: self.P(t))
    def bump_key_rate(self, tenor: float, bump_bp: float) -> "PiecewiseFlatZeroCurve":
        bump = bump_bp*1e-4
        newz = self.zeros.copy()
        idx = int(np.argmin(np.abs(self.tenors - tenor)))
        newz[idx] += bump
        return PiecewiseFlatZeroCurve(self.tenors.tolist(), newz.tolist())

def curve_from_pillars(tenors: List[float], discounts_or_zeros: List[float], input_type: str = "discount") -> DiscountCurve:
    ten = np.array(tenors, float); vals = np.array(discounts_or_zeros, float)
    idx = np.argsort(ten); ten = ten[idx]; vals = vals[idx]
    if input_type == "discount":
        def df(t):
            return float(np.interp(t, ten, vals, left=vals[0], right=vals[-1]))
        return DiscountCurve(df)
    elif input_type == "zero_cont":
        def df(t):
            z = float(np.interp(t, ten, vals, left=vals[0], right=vals[-1]))
            return math.exp(-z*t)
        return DiscountCurve(df)
    else:
        raise ValueError("input_type must be 'discount' or 'zero_cont'")

# ===================== Futures & Black-76 =====================

def implied_avg_rate_from_curve(curve: DiscountCurve, t_start: float, t_end: float) -> float:
    return 100.0 * curve.fwd_simple(t_start, t_end)

def sofr_futures_price(avg_rate_exp_pct: float) -> float:
    return 100.0 - avg_rate_exp_pct

def futures_convexity_adjustment(sigma: float, t: float, T: float) -> float:
    return 0.5*(sigma**2)*t*T

@dataclass
class Black76Result:
    price: float
    delta: float
    gamma: float
    vega: float
    d1: float
    d2: float

def black76_option_on_futures(F: float, K: float, T: float, sigma: float, is_call: bool, df_settlement: float = 1.0) -> Black76Result:
    if F<=0 or K<=0 or sigma<=0 or T<=0:
        return Black76Result(0.0,0.0,0.0,0.0,float("nan"),float("nan"))
    vT = sigma*math.sqrt(T)
    d1 = (math.log(F/K) + 0.5*sigma*sigma*T)/vT
    d2 = d1 - vT
    if is_call:
        price = df_settlement*(F*norm_cdf(d1) - K*norm_cdf(d2))
        delta = df_settlement*norm_cdf(d1)
    else:
        price = df_settlement*(K*norm_cdf(-d2) - F*norm_cdf(-d1))
        delta = -df_settlement*norm_cdf(-d1)
    gamma = df_settlement*norm_pdf(d1)/(F*vT)
    vega  = df_settlement*F*norm_pdf(d1)*math.sqrt(T)
    return Black76Result(price, delta, gamma, vega, d1, d2)

# ===================== Swaps =====================

@dataclass
class SwapAnalytics:
    par_rate: float
    pv_fixed: float
    pv_float: float
    pv: float
    annuity: float
    dv01: float

def swap_par_rate(curve: DiscountCurve, pay_times: List[float], accruals: List[float]) -> SwapAnalytics:
    if len(pay_times)!=len(accruals):
        raise ValueError("pay_times and accruals mismatch")
    discounts = np.array([curve.P(t) for t in pay_times])
    annuity = float(np.sum(np.array(accruals)*discounts))
    Pn = curve.P(pay_times[-1])
    k_star = safe_div(1.0 - Pn, annuity)
    pv_fixed = k_star*annuity
    pv_float = 1.0 - Pn
    pv = pv_fixed - pv_float
    dv01 = annuity*1e-4
    return SwapAnalytics(k_star, pv_fixed, pv_float, pv, annuity, dv01)

def swap_pv_given_rate(curve: DiscountCurve, K: float, pay_times: List[float], accruals: List[float]) -> float:
    discounts = np.array([curve.P(t) for t in pay_times])
    annuity = float(np.sum(np.array(accruals)*discounts))
    Pn = curve.P(pay_times[-1])
    return K*annuity - (1.0 - Pn)

def swap_bucketed_cashflow_dv01(curve: DiscountCurve, pay_times: List[float], accruals: List[float]) -> List[Tuple[float,float]]:
    return [(t, a*curve.P(t)*1e-4) for t,a in zip(pay_times, accruals)]

def swap_parallel_delta(curve: DiscountCurve, K: float, pay_times: List[float], accruals: List[float], bump_bp: float = 1.0) -> float:
    bump = bump_bp*1e-4
    def bumped_P(t: float) -> float:
        r = curve.zero_rate_cont(t)
        return math.exp(-(r+bump)*t)
    bumped_curve = DiscountCurve(bumped_P)
    base_pv = swap_pv_given_rate(curve, K, pay_times, accruals)
    bumped_pv = swap_pv_given_rate(bumped_curve, K, pay_times, accruals)
    return bumped_pv - base_pv

# ===================== Caps/Floors =====================

@dataclass
class CapletResult:
    price: float
    delta_fwd: float
    gamma_fwd: float
    vega: float
    d1: float
    d2: float

def black_caplet_price(df_start: float, df_end: float, T_start: float, T_end: float, K: float, sigma: float, is_caplet: bool, accrual: Optional[float]=None) -> CapletResult:
    tau = accrual if accrual is not None else max(T_end - T_start, 1e-12)
    L = (df_start/df_end - 1.0)/tau
    T_expiry = T_start
    if L<=0 or K<=0 or sigma<=0 or T_expiry<=0:
        return CapletResult(0.0,0.0,0.0,0.0,float("nan"),float("nan"))
    vT = sigma*math.sqrt(T_expiry)
    d1 = (math.log(L/K) + 0.5*sigma*sigma*T_expiry)/vT
    d2 = d1 - vT
    df_pay = df_end
    if is_caplet:
        price = df_pay * tau * (L*norm_cdf(d1) - K*norm_cdf(d2))
        delta_fwd = df_pay * tau * norm_cdf(d1)
    else:
        price = df_pay * tau * (K*norm_cdf(-d2) - L*norm_cdf(-d1))
        delta_fwd = -df_pay * tau * norm_cdf(-d1)
    gamma_fwd = df_pay * tau * norm_pdf(d1)/(L*vT)
    vega = df_pay * tau * L * norm_pdf(d1) * math.sqrt(T_expiry)
    return CapletResult(price, delta_fwd, gamma_fwd, vega, d1, d2)

def black_cap_floor(curve: DiscountCurve, schedule: List[Tuple[float,float,float]], K: float, sigmas: List[float], is_cap: bool) -> Dict[str, float]:
    if len(schedule)!=len(sigmas):
        raise ValueError("schedule and sigmas must be same length")
    total_price = 0.0; delta_sum = 0.0; gamma_sum = 0.0; vega_sum = 0.0
    for (T0,T1,tau), vol in zip(schedule, sigmas):
        df0, df1 = curve.P(T0), curve.P(T1)
        res = black_caplet_price(df0, df1, T0, T1, K, vol, is_cap, accrual=tau)
        total_price += res.price
        delta_sum  += res.delta_fwd
        gamma_sum  += res.gamma_fwd
        vega_sum   += res.vega
    return {"price": total_price, "delta_fwd_sum": delta_sum, "gamma_fwd_sum": gamma_sum, "vega_sum": vega_sum}

def cap_bucketed_vega(curve: DiscountCurve, schedule: List[Tuple[float,float,float]], K: float, sigmas: List[float], is_cap: bool) -> List[Tuple[float,float]]:
    out = []
    for (T0,T1,tau), vol in zip(schedule, sigmas):
        df0, df1 = curve.P(T0), curve.P(T1)
        res = black_caplet_price(df0, df1, T0, T1, K, vol, is_cap, accrual=tau)
        out.append((T1, res.vega))
    return out

# ===================== Roll-down & Carry =====================

def roll_down_forward(curve: DiscountCurve, T1: float, T2: float) -> float:
    return curve.fwd_simple(T1, T2)

def carry_pnl_instrument(pv_func: Callable[[float], float], dt: float = 1.0/252) -> float:
    return pv_func(dt) - pv_func(0.0)

def swap_pv_with_horizon(curve: DiscountCurve, K: float, pay_times: List[float], accruals: List[float]) -> Callable[[float], float]:
    base = list(zip(pay_times, accruals))
    def pv_given_h(h: float) -> float:
        new_times = [max(0.0, t-h) for t,_ in base if t-h>1e-8]
        new_accrs = [a for (t,a) in base if t-h>1e-8]
        if not new_times:
            return 0.0
        return swap_pv_given_rate(curve, K, new_times, new_accrs)
    return pv_given_h

@dataclass
class ThetaBreakdown:
    total_theta: float
    fixed_accrual: float
    float_accrual: float
    residual_roll: float

def swap_theta_breakdown(curve: DiscountCurve, K: float, pay_times: List[float], accruals: List[float], dt_years: float = 1.0/252) -> ThetaBreakdown:
    pv_fn = swap_pv_with_horizon(curve, K, pay_times, accruals)
    total = carry_pnl_instrument(pv_fn, dt_years)
    fixed_accrual = K*dt_years
    r_short = curve.zero_rate_cont(max(1e-6, 0.5*dt_years))
    float_accrual = r_short*dt_years
    residual = total - fixed_accrual - float_accrual
    return ThetaBreakdown(total, fixed_accrual, float_accrual, residual)

# ===================== Schedules (calendar-aware) =====================

def add_months(d: dt.date, n: int) -> dt.date:
    y = d.year + (d.month - 1 + n)//12
    m = (d.month - 1 + n)%12 + 1
    # month end handling
    if m in (1,3,5,7,9,11):
        last_day = 31
    elif m in (4,6,8,10,12):
        last_day = 30 if m in (4,6,9,11) else 31
    else:
        # February
        last_day = 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28
    return dt.date(y, m, min(d.day, last_day))

def gen_swap_schedule(start: dt.date, end: dt.date, freq_months: int, cal: BusinessCalendar, adj: str = "following", fixed_basis: str = "30/360") -> Tuple[List[dt.date], List[float]]:
    dates = []
    d = start
    while True:
        d = add_months(d, freq_months)
        if d >= end:
            dates.append(end); break
        dates.append(d)
    pay_dates = [cal.adjust(dd, adj) for dd in dates]
    prev = start
    accr = []
    for pd in pay_dates:
        accr.append(yf(prev, pd, fixed_basis))
        prev = pd
    return pay_dates, accr

def dates_to_times(valuation: dt.date, dates: List[dt.date], basis: str) -> List[float]:
    return [yf(valuation, d, basis) for d in dates]

# ===================== Key-Rate PV01 report =====================

def krd_pv01_report(curve_pwz: "PiecewiseFlatZeroCurve",
                    price_fn: Callable[[DiscountCurve], float],
                    key_tenors: List[float],
                    bump_bp: float = 1.0):
    base_dc = curve_pwz.to_discount_curve()
    base_pv = price_fn(base_dc)
    rows = []
    for T in key_tenors:
        bumped = curve_pwz.bump_key_rate(T, bump_bp).to_discount_curve()
        pv_b = price_fn(bumped)
        rows.append({"tenor": T, "base_pv": base_pv, "bumped_pv": pv_b, "dpv": pv_b - base_pv, "bump_bp": bump_bp})
    if pd is None:
        return rows  # fallback list of dicts
    return pd.DataFrame(rows)

# ===================== __main__ demo =====================


# ======== Exchange-style US market holidays & early closes (simple) ========

def nyse_market_holidays(year:int)->set:
    """Approximate NYSE full-day market holidays, including Good Friday."""
    H = us_fed_holidays(year)
    # Good Friday: two days before Easter Sunday (approx via Anonymous Gregorian algorithm)
    H.add(approx_good_friday(year))
    return H

def approx_good_friday(year:int)->dt.date:
    """Meeus/Jones/Butcher algorithm for Easter; Good Friday is Easter-2 days."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    easter = dt.date(year, month, day)
    return easter - dt.timedelta(days=2)

class MarketCalendar(BusinessCalendar):
    """BusinessCalendar with early-close awareness (flag only; no intraday modeling here)."""
    def __init__(self, holidays: Optional[set] = None, early_closes: Optional[set] = None):
        super().__init__(holidays=holidays)
        self.early_closes = early_closes or set()
    def is_early_close(self, d: dt.date) -> bool:
        return d in self.early_closes

def nyse_calendar_for_years(start_year:int, end_year:int)->MarketCalendar:
    hol = set()
    for y in range(start_year, end_year+1):
        hol |= nyse_market_holidays(y)
    # Simple early-closes (half-day): Day after Thanksgiving, Christmas Eve (if weekday)
    ec = set()
    for y in range(start_year, end_year+1):
        # Day after Thanksgiving
        thanks = nth_weekday_of_month(y, 11, 3, 4)  # Thu
        fri = thanks + dt.timedelta(days=1)
        if fri.weekday() < 5: ec.add(fri)
        # Christmas Eve
        xmas_eve = dt.date(y,12,24)
        if xmas_eve.weekday() < 5 and xmas_eve not in hol:
            ec.add(xmas_eve)
    return MarketCalendar(holidays=hol, early_closes=ec)

# ======== IMM pack/bundle DV01 bucketing for swap cashflows ========

def dv01_by_imm_pack(pay_dates: List[dt.date],
                     accruals: List[float],
                     curve: DiscountCurve,
                     valuation_date: dt.date,
                     start_pack: Optional[dt.date] = None) -> Dict[str, float]:
    """
    Bucket per-cashflow PV01 into IMM packs by calendar date:
      White (yr1), Red (yr2), Green (yr3), Blue (yr4), Gold (yr5), etc.
    """
    if len(pay_dates)!=len(accruals):
        raise ValueError("pay_dates and accruals mismatch")
    if start_pack is None:
        start_pack = next_imm(valuation_date)
    # Build pack boundaries: 6 years (24 quarters)
    packs = imm_packs(start_pack, years=6)  # list of lists of 4 IMMs
    boundaries = [packs[0][0]]  # start
    for p in packs:
        boundaries.append(p[-1])  # end of each pack
    # Map date -> pack index
    def pack_index(d: dt.date)->int:
        idx = 0
        for k in range(len(packs)):
            start_k = packs[k][0]
            end_k = packs[k][-1]
            if d <= end_k:
                return k
        return len(packs)-1  # beyond -> last
    name = ["White","Red","Green","Blue","Gold","Purple"]
    out: Dict[str, float] = {}
    for pd, a in zip(pay_dates, accruals):
        t = yf(valuation_date, pd, "ACT/365F")
        pv01 = a*curve.P(t)*1e-4
        k = pack_index(pd)
        label = name[k] if k < len(name) else f"Pack{k+1}"
        out[label] = out.get(label, 0.0) + pv01
    return out

def dv01_by_imm_bundle(pay_dates: List[dt.date],
                       accruals: List[float],
                       curve: DiscountCurve,
                       valuation_date: dt.date,
                       bundles: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Aggregate pack DV01s into user-defined bundles.
    Example bundles: {"2y bundle":["White","Red","Green","Blue"]}
    """
    pack_dv01 = dv01_by_imm_pack(pay_dates, accruals, curve, valuation_date)
    out: Dict[str, float] = {}
    for bname, pack_list in bundles.items():
        out[bname] = sum(pack_dv01.get(p, 0.0) for p in pack_list)
    return out

# ======== Simple OIS curve bootstrap from par OIS quotes ========

def bootstrap_ois_curve(valuation_date: dt.date,
                        maturities_years: List[float],
                        par_rates: List[float],
                        fixed_freq_per_year: int = 2,
                        fixed_basis: str = "30/360",
                        time_basis: str = "ACT/365F",
                        cal: BusinessCalendar = None) -> DiscountCurve:
    """
    Very simple bootstrap:
      - For each maturity T_n, create a fixed-leg schedule from valuation_date to T_n
        with freq=fixed_freq_per_year and accruals by fixed_basis.
      - Solve for P(0,T_n) using par swap identity:
           K * sum_{i=1}^{n} α_i P(0,T_i) = 1 - P(0,T_n)
        where all P(0,T_i) for i<n are already bootstrapped.
      - Assumes each T_n lines up exactly on the schedule grid (no stubs).
    Returns a DiscountCurve using linear interpolation in ln(P).
    """
    assert len(maturities_years)==len(par_rates)>=1
    cal = cal or nyc_calendar_for_years(valuation_date.year-1, valuation_date.year+10)
    # Container for nodes
    nodes_times: List[float] = []
    nodes_P: List[float] = []
    for T, K in zip(maturities_years, par_rates):
        end_date = valuation_date + dt.timedelta(days=int(T*365.25))
        # build schedule
        months = int(round(12*T))
        step = int(round(12/fixed_freq_per_year))
        # use month-based schedule generator to reduce day-count drift
        start = valuation_date
        # generate payment dates
        def add_months(d: dt.date, n: int)->dt.date:
            y = d.year + (d.month - 1 + n)//12
            m = (d.month - 1 + n)%12 + 1
            # month end handling
            last_day = [31, 29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m-1]
            return dt.date(y, m, min(d.day, last_day))
        # build up to maturity
        pay_dates = []
        d = start
        while True:
            d = add_months(d, step)
            if d >= add_months(start, months):
                pay_dates.append(add_months(start, months)); break
            pay_dates.append(d)
        pay_dates = [cal.adjust(x, "following") for x in pay_dates]
        accr = []
        prev = start
        for pd in pay_dates:
            accr.append(yf(prev, pd, fixed_basis))
            prev = pd
        pay_times = [yf(valuation_date, pd, time_basis) for pd in pay_dates]
        # compute sum α_i P(0,T_i) with P known for earlier nodes
        S = 0.0
        for t, a in zip(pay_times[:-1], accr[:-1]):
            # interpolate P(t) from existing nodes (ln P linear)
            if not nodes_times:
                raise ValueError("First maturity must be the first coupon date; provide shortest tenor first.")
            P = float(np.interp(t, nodes_times, nodes_P)) if len(nodes_times)>1 else nodes_P[0]
            S += a * P
        # Solve for P(T_n) (last payment)
        a_n = accr[-1]
        t_n = pay_times[-1]
        Pn = (1.0 - K * S) / (1.0 + K * a_n)
        nodes_times.append(t_n)
        nodes_P.append(Pn)
    # build discount curve with ln(P) linear interp
    def df(t: float)->float:
        if t <= nodes_times[0]:
            return nodes_P[0]
        if t >= nodes_times[-1]:
            return nodes_P[-1]
        # linear in ln(P)
        lnP = np.interp(t, nodes_times, np.log(np.maximum(nodes_P, 1e-300)))
        return float(np.exp(lnP))
    return DiscountCurve(df)

if __name__ == "__main__":

    curve = DiscountCurve.from_flat_rate(0.04, "cont")
    # Futures 6M->9M
    t0, t1 = 0.5, 0.75
    avg_pct = implied_avg_rate_from_curve(curve, t0, t1)
    print("Futures expected avg %:", avg_pct, " price:", sofr_futures_price(avg_pct))

    # IMM mapping
    today = dt.date.today()
    nxt = next_imm(today)
    print("Next IMM:", nxt, " SR3 code:", cme_sr3_code_from_imm(nxt))

    # Calendar-aware 5Y swap schedule
    NYC = nyc_calendar_for_years(today.year-1, today.year+5)
    start = today
    end = add_months(today, 60)
    pay_dates, accr = gen_swap_schedule(start, end, 6, NYC, "following", "30/360")
    pay_times = dates_to_times(today, pay_dates, "ACT/365F")
    sa = swap_par_rate(curve, pay_times, accr)
    print("5Y par:", sa.par_rate, "annuity:", sa.annuity, "DV01:", sa.dv01)

    # KRD report
    pwz = PiecewiseFlatZeroCurve([0.5,1,2,3,5,7,10],[0.040,0.041,0.042,0.043,0.044,0.045,0.046])
    def price_fn(dc: DiscountCurve):
        return swap_pv_given_rate(dc, sa.par_rate, pay_times, accr)
    krd_df = krd_pv01_report(pwz, price_fn, [0.5,1,2,3,5,7,10], bump_bp=1.0)
    print("KRD rows:", krd_df if pd is None else krd_df.to_string(index=False))
