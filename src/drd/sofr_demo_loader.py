
import os

import pandas as pd
import numpy as np

import datetime as dt

from sofr_pricing_lib import (
    nyc_calendar_for_years,
    bootstrap_ois_curve,
    implied_avg_rate_from_curve,
    sofr_futures_price,
    black76_option_on_futures,
    cme_sr3_code_from_imm,
    next_imm,
    sofr_futures_3m_window,
    yf,
    DiscountCurve,
    swap_par_rate,
    swap_pv_given_rate,
    gen_swap_schedule,
    dates_to_times,
    black_caplet_price
)


VAL_DATE = dt.date.fromisoformat('2025-09-20')


# === Load OIS quotes and bootstrap curve ===
ois = pd.read_csv('ois_par_quotes.csv', parse_dates=['valuation_date'])
curve = bootstrap_ois_curve(
    valuation_date=VAL_DATE,
    maturities_years=ois['maturity_years'].tolist(),
    par_rates=ois['par_rate_decimal'].tolist(),
    fixed_freq_per_year=int(ois['fixed_freq_per_year'].iloc[0]),
    fixed_basis=str(ois['fixed_daycount'].iloc[0]),
    time_basis=str(ois['time_basis'].iloc[0]),
    cal=nyc_calendar_for_years(VAL_DATE.year-1, VAL_DATE.year+10)
)

# === Load futures and recompute prices (verification) ===
fut = pd.read_csv('sofr_futures.csv', parse_dates=['valuation_date','imm_date','start_date','end_date'])
def recompute_fut(row):
    T0 = yf(VAL_DATE, row['start_date'].date(), 'ACT/365F')
    T1 = yf(VAL_DATE, row['end_date'].date(), 'ACT/365F')
    fwd = curve.fwd_simple(T0, T1)
    px  = sofr_futures_price(100.0*fwd)
    return pd.Series({'re_fwd': fwd, 're_px': px})
fut[['re_fwd','re_px']] = fut.apply(recompute_fut, axis=1)
print('Futures recompute check:\n', fut[['contract_code','implied_avg_rate_decimal','re_fwd','price','re_px']])

# === Load futures options and recompute one ===
opt = pd.read_csv('sofr_futures_options.csv', parse_dates=['valuation_date'])
row0 = opt.iloc[0]
res = black76_option_on_futures(row0['underlying_forward_decimal'], row0['strike_decimal'],
                                row0['expiry_T_years'], row0['sigma_annual'], True, row0['df_settlement'])
print('\nOption sample recompute (call):', res.price, 'delta:', res.delta)

# === Load swaps and recompute PV ===
swaps = pd.read_csv('swap_trades.csv', parse_dates=['valuation_date'])
for idx, r in swaps.iterrows():
    end_date = VAL_DATE + dt.timedelta(days=int(365.25*r['tenor_years']))
    step = int(round(12/r['fixed_freq_per_year']))
    pay_dates, accr = gen_swap_schedule(VAL_DATE, end_date, step, nyc_calendar_for_years(VAL_DATE.year-1, VAL_DATE.year+10), 'following', r['fixed_daycount'])
    pay_times = dates_to_times(VAL_DATE, pay_dates, 'ACT/365F')
    pv = swap_pv_given_rate(curve, r['fixed_rate_decimal'], pay_times, accr)
    swaps.loc[idx, 'pv_recomputed'] = pv
print('\nSwaps PV recompute check:\n', swaps[['tenor_years','fixed_rate_decimal','pv_per_1_notional','pv_recomputed']])

# === Load cap surface and verify one caplet ===
caps = pd.read_csv('cap_surface_quotes.csv', parse_dates=['valuation_date'])
c0 = caps.iloc[0]
df0 = curve.P(c0['T_start']); df1 = curve.P(c0['T_end'])
res_caplet = black_caplet_price(df0, df1, c0['T_start'], c0['T_end'], c0['strike_decimal'], c0['sigma_annual'], True, c0['accrual'])
print('\nCaplet sample recompute price:', res_caplet.price)
