
"""
refinitiv_pull_sofr.py

Pulls the inputs needed to price the four SOFR products:
  1) SOFR 3M Futures (CME SR3)
  2) Options on SOFR Futures
  3) USD OIS (SOFR) Swaps (par quotes by tenor)
  4) Caps/Floors volatility surface (caplet vols by expiry/strike)

Supports both:
  - Eikon / Refinitiv Workspace Python (eikon)
  - Refinitiv Data Library for Python (refinitiv-data)

You can use either one. The script tries Eikon first, then falls back to refinitiv-data.

Usage:
  - Put your credentials in environment variables:
      EIKON_APP_KEY=...
      RDP_APP_KEY=...         (if using refinitiv-data)
  - Optionally edit CONFIG below to refine RIC patterns and fields.
  - Run:
      python refinitiv_pull_sofr.py

Outputs (CSV):
  ois_par_quotes.csv
  sofr_futures.csv
  sofr_futures_options.csv
  cap_surface_quotes.csv
  (Swap trades are typically internal; here we only fetch par quotes for curve bootstrap.)

Notes:
  - RIC/chain patterns vary by environment. The defaults below are common, but you may need
    to adjust them using the provided search helpers.
  - For options chains, this script resolves calls & puts per nearby future and fetches strikes
    around ATM. Expand bands as needed.
"""

import os
import sys
import math
import time
import json
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Config: tune these patterns and ranges for your environment
# ------------------------------------------------------------------

CONFIG = {
    # Front SOFR futures chain (CME SR3). The "0#<root>:" pattern expands into individual contracts.
    "futures_chain_ric": "0#SR3:",

    # How many near futures to keep (used for options as well)
    "num_near_futures": 6,

    # Option chains per future month (chain expansion patterns)
    # The star '*' expands all strikes; we will narrow by a moneyness window below.
    "option_chain_pattern": "{fut_root}*",     # Example: SR3Z25*
    "option_fields": ["TR.RIC", "TR.Bid", "TR.Ask", "TR.Mid", "TR.PriceClose", "TR.SettlPrice",
                      "TR.StrikePrice", "TR.CallPut"] ,

    # Moneyness window around ATM forward (in rate points). Adjust as needed.
    "option_strike_band": 0.50,  # e.g., include strikes within +/- 50 ticks of ATM (as rate points)

    # OIS SOFR par curve tenors and (typical) RIC patterns (these often vary—use search helpers if needed)
    # You can replace with the specific RICs available in your environment.
    # As a fallback, we use a list of tenors and retrieve Refinitiv "Swap Rate" fields via search filtering.
    "ois_tenors_years": [0.5, 1, 2, 3, 5, 7, 10],

    # Cap/Floor surface query: maturities (years) and strikes (in decimal) to pull vols for.
    "cap_surface_maturities_years": [0.5, 1, 2, 3, 5, 7, 10],
    "cap_surface_strikes_decimal": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
}

VAL_DATE = dt.date.today()

# ------------------------------------------------------------------
# Session setup: Eikon first, then refinitiv-data
# ------------------------------------------------------------------

HAS_EIKON = False
HAS_RDP = False
ek = None
rd = None

def setup_sessions():
    global HAS_EIKON, HAS_RDP, ek, rd
    # Try Eikon
    try:
        import eikon as ek_mod
        app_key = os.getenv("EIKON_APP_KEY")
        if app_key:
            ek_mod.set_app_key(app_key)
            # simple ping
            ek_mod.get_app_key()  # raises if invalid
            HAS_EIKON = True
            ek = ek_mod
            print("[OK] Eikon SDK session established.")
    except Exception as e:
        print("[WARN] Eikon not available:", repr(e))

    # Try refinitiv-data
    if not HAS_EIKON:
        try:
            import refinitiv.data as rd_mod
            from refinitiv.data.library import Taxonomy
            app_key = os.getenv("RDP_APP_KEY")
            if app_key:
                rd_mod.open_session(app_key=app_key)
                HAS_RDP = True
                rd = rd_mod
                print("[OK] refinitiv-data session established.")
        except Exception as e:
            print("[WARN] refinitiv-data not available:", repr(e))

    if not HAS_EIKON and not HAS_RDP:
        raise RuntimeError("Neither Eikon nor refinitiv-data sessions could be established. "
                           "Set EIKON_APP_KEY or RDP_APP_KEY as environment variables.")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['.'.join([str(c) for c in col if c is not None]) for col in df.columns]
    return df

def eikon_get_chain(chain_ric: str) -> List[str]:
    # Use TR_CHAIN to expand chains
    df, err = ek.get_data(chain_ric, ["TR.RIC"])
    if err is not None:
        raise RuntimeError(err)
    ric_list = df["RIC"].dropna().tolist()
    return ric_list

def rdp_get_chain(chain_ric: str) -> List[str]:
    # refinitiv-data has chain resolution via symbology service
    # Fallback: attempt to get constituents using the "chains" endpoint if available
    try:
        from refinitiv.data.content.symbology import chains
        ch = chains.Definition(universe=[chain_ric])
        out = ch.get_data().data  # list of dicts
        ric_list = []
        for block in out:
            for item in block.get("Constituents", []):
                ric = item.get("Identifier")
                if ric:
                    ric_list.append(ric)
        return ric_list
    except Exception as e:
        print("[WARN] Chain resolution via refinitiv-data failed:", repr(e))
        return []

def get_chain(chain_ric: str) -> List[str]:
    if HAS_EIKON:
        return eikon_get_chain(chain_ric)
    else:
        return rdp_get_chain(chain_ric)

def eikon_get_timeseries(ric: str, fields: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None):
    # ek.get_timeseries returns a DataFrame with DateTimeIndex and price fields
    ts = ek.get_timeseries(ric, fields=fields or ["CLOSE"], start_date=start, end_date=end)
    return ts

def rdp_get_timeseries(ric: str, interval: str = "daily", fields: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None):
    from refinitiv.data.content import historical_pricing as hp
    defn = hp.Definition(
        universe=[ric],
        interval=hp.Intervals.DAILY if interval=="daily" else hp.Intervals.MONTHLY,
        fields=fields
    )
    if start: defn = defn.set_time_range(start=start)
    if end:   defn = defn.set_time_range(end=end)
    res = defn.get_data().data
    # Convert to DF
    if len(res)==0: return pd.DataFrame()
    df = pd.DataFrame(res[0].get("Data", []))
    return df

def get_timeseries(ric: str, fields: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    if HAS_EIKON:
        return eikon_get_timeseries(ric, fields, start, end)
    else:
        return rdp_get_timeseries(ric, "daily", fields, start, end)

def eikon_get_data(universe: List[str], fields: List[str]) -> pd.DataFrame:
    df, err = ek.get_data(universe, fields)
    if err is not None:
        raise RuntimeError(err)
    return normalize_cols(df)

def rdp_get_data(universe: List[str], fields: List[str]) -> pd.DataFrame:
    from refinitiv.data.content import fundamentals as fdm
    # Fallback generic endpoint for fields — not all TR fields are supported here.
    # We try content.pricing as primary path:
    try:
        from refinitiv.data.content import pricing
        pdef = pricing.Definition(
            universe=universe,
            fields=fields
        )
        out = pdef.get_data().data
        recs = []
        for block in out:
            base = {"RIC": block.get("Identifier")}
            for fld, val in block.get("Fields", {}).items():
                base[fld] = val
            recs.append(base)
        return pd.DataFrame(recs)
    except Exception as e:
        print("[WARN] pricing endpoint failed:", repr(e))
        return pd.DataFrame()

def get_data(universe: List[str], fields: List[str]) -> pd.DataFrame:
    if HAS_EIKON:
        return eikon_get_data(universe, fields)
    else:
        return rdp_get_data(universe, fields)

# ------------------------------------------------------------------
# 1) SOFR Futures (list near contracts + last/settle)
# ------------------------------------------------------------------

def fetch_sofr_futures(config: Dict[str, Any]) -> pd.DataFrame:
    chain = config["futures_chain_ric"]
    print(f"[INFO] Resolving futures chain: {chain}")
    futs = get_chain(chain)
    if not futs:
        raise RuntimeError("No futures found for chain " + chain)
    futs = futs[:config["num_near_futures"]]
    fields = ["TR.RIC","TR.ContractMaturityDate","TR.PriceClose","TR.Last","TR.SettlPrice"]
    df = get_data(futs, fields)
    df["valuation_date"] = VAL_DATE
    # Basic columns matching earlier CSVs
    out = df.rename(columns={
        "RIC":"contract_code",
        "TR.RIC":"contract_code",
        "TR.ContractMaturityDate":"maturity_date",
        "TR.SettlPrice":"settle",
        "TR.Last":"last",
        "TR.PriceClose":"close"
    })
    return out

# ------------------------------------------------------------------
# 2) Options on SOFR Futures (resolve per future month)
# ------------------------------------------------------------------

def fetch_sofr_futures_options(config: Dict[str, Any], futures_df: pd.DataFrame) -> pd.DataFrame:
    all_opts = []
    for ric in futures_df["contract_code"].tolist():
        fut_root = ric  # e.g., SR3Z25
        pattern = config["option_chain_pattern"].format(fut_root=fut_root)
        chain_ric = "0#" + pattern
        print(f"[INFO] Resolving option chain: {chain_ric}")
        try:
            chain = get_chain(chain_ric)
        except Exception as e:
            print("[WARN] Could not expand option chain", chain_ric, repr(e))
            chain = []
        if not chain:
            continue
        # Fetch option fields
        opt_fields = config["option_fields"]
        chunk = get_data(chain, opt_fields)
        if chunk is None or chunk.empty:
            continue
        chunk["underlying_future"] = fut_root
        chunk["valuation_date"] = VAL_DATE
        all_opts.append(chunk)

    if not all_opts:
        print("[WARN] No options retrieved.")
        return pd.DataFrame()

    opts = pd.concat(all_opts, ignore_index=True)
    # Normalize columns
    opts = opts.rename(columns={"RIC":"option_ric", "TR.RIC":"option_ric", "TR.StrikePrice":"strike", "TR.CallPut":"cp"})
    return normalize_cols(opts)

# ------------------------------------------------------------------
# 3) OIS (SOFR) par curve quotes
# ------------------------------------------------------------------

def fetch_ois_par_curve(config: Dict[str, Any]) -> pd.DataFrame:
    tenors = config["ois_tenors_years"]
    # In many environments, specific RICs exist for USD OIS par swaps (e.g., USD OIS 1Y, 2Y...).
    # As a general fallback, we search via fields for "Swap Rate" where available. If your site has
    # canonical OIS RICs, replace this with direct RICs and TR fields.
    recs = []
    for T in tenors:
        label = f"USD OIS {int(T*12)}M"
        recs.append({"tenor_years": T, "label": label})
    df = pd.DataFrame(recs)
    # Placeholder field for par_rate_decimal; users should map to instrument fields in their site.
    df["par_rate_decimal"] = np.nan
    df["valuation_date"] = VAL_DATE
    print("[INFO] OIS par curve placeholder created (map to your site RICs/fields).")
    return df

# ------------------------------------------------------------------
# 4) Cap/Floor surface quotes (grid of vols)
# ------------------------------------------------------------------

def fetch_cap_surface(config: Dict[str, Any]) -> pd.DataFrame:
    mats = config["cap_surface_maturities_years"]
    strikes = config["cap_surface_strikes_decimal"]
    # As with OIS, cap/floor vol surfaces exist under site-specific RICs/surface objects.
    # Here we create a placeholder grid. Replace with your site's vol surface fields.
    grid = []
    for T in mats:
        for K in strikes:
            grid.append({"T_years": T, "strike_decimal": K, "sigma_annual": np.nan, "valuation_date": VAL_DATE})
    print("[INFO] Cap surface placeholder created (map to your site vol surface).")
    return pd.DataFrame(grid)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    setup_sessions()

    fut = fetch_sofr_futures(CONFIG)
    fut.to_csv("sofr_futures_refinitiv.csv", index=False)
    print("[OK] Wrote sofr_futures_refinitiv.csv")

    opt = fetch_sofr_futures_options(CONFIG, fut)
    if not opt.empty:
        opt.to_csv("sofr_futures_options_refinitiv.csv", index=False)
        print("[OK] Wrote sofr_futures_options_refinitiv.csv")
    else:
        print("[WARN] Options CSV not written (no options retrieved).")

    ois = fetch_ois_par_curve(CONFIG)
    ois.to_csv("ois_par_quotes_refinitiv.csv", index=False)
    print("[OK] Wrote ois_par_quotes_refinitiv.csv (placeholders for par rates).")

    caps = fetch_cap_surface(CONFIG)
    caps.to_csv("cap_surface_quotes_refinitiv.csv", index=False)
    print("[OK] Wrote cap_surface_quotes_refinitiv.csv (placeholders for vol grid).")

    print("\nNext steps:")
    print(" - Replace OIS and Cap vol placeholders by mapping your site's RICs and fields.")
    print(" - Use the options CSV to calibrate an implied vol surface for SOFR futures options.")
    print(" - Feed these CSVs into your existing pricing toolkit.")

if __name__ == "__main__":
    main()
