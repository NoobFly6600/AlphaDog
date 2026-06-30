"""
factor_analysis.py
因子IC分析：数据准备 + 因子质量评估
"""

import pandas as pd
import numpy as np
import sys
import time
sys.path.append('..')
from data.fetcher import (
    init, get_daily_basic, get_moneyflow, get_margin,
    get_price_on_date, get_industry_map, get_cached_financials,
    get_financials_as_of, get_trade_cal
)
from model.factor import get_market_state, calculate_momentum_fast, calculate_factors
from datetime import datetime, timedelta

ANALYSIS_START = '20230101'
ANALYSIS_END   = '20241231'
FORWARD_DAYS   = [5, 10, 20]
SAMPLE_FREQ    = 21  # 每月取一个截面
FACTOR_COLS    = ['pe_score', 'pb_score', 'momentum_score',
                  'moneyflow_score', 'roe_score', 'margin_score']


# ===============================
# 1. IC / RankIC
# ===============================

def calc_ic(df, factor_col, forward_return_col='fwd_ret'):
    valid = df[[factor_col, forward_return_col]].dropna()
    if len(valid) < 50:
        return np.nan
    return valid[factor_col].corr(valid[forward_return_col])


def calc_rank_ic(df, factor_col, forward_return_col='fwd_ret'):
    valid = df[[factor_col, forward_return_col]].dropna()
    if len(valid) < 50:
        return np.nan
    return valid[factor_col].rank().corr(valid[forward_return_col].rank())


# ===============================
# 2. 时间稳定性
# ===============================

def rolling_ic(df, factor_col, forward_return_col='fwd_ret', window=5):
    df = df.copy().sort_values('date')
    dates = df['date'].unique()
    ic_list = []
    for i in range(len(dates) - window):
        subset = df[df['date'].isin(dates[i:i + window])]
        ic = calc_rank_ic(subset, factor_col, forward_return_col)
        ic_list.append(ic)
    return ic_list


def print_ic_stability(ic_list, factor_name):
    ic_arr = np.array([x for x in ic_list if not np.isnan(x)])
    if len(ic_arr) == 0:
        print(f"  {factor_name}: 无数据")
        return
    print(f"  均值:{np.mean(ic_arr):.4f}  标准差:{np.std(ic_arr):.4f}  "
          f"胜率:{np.mean(ic_arr > 0):.0%}  "
          f"IC_IR:{np.mean(ic_arr)/np.std(ic_arr):.3f}" if np.std(ic_arr) > 0 else "")


# ===============================
# 3. 单因子 / 批量分析
# ===============================

def analyze_factor(df, factor_col, forward_return_col='fwd_ret'):
    ic      = calc_ic(df, factor_col, forward_return_col)
    rank_ic = calc_rank_ic(df, factor_col, forward_return_col)
    print(f"\n  {factor_col}")
    print(f"    IC={ic:.4f}  RankIC={rank_ic:.4f}" if not np.isnan(ic) else "    IC=NaN")
    if not np.isnan(rank_ic):
        if abs(rank_ic) < 0.01:
            print("    结论: 几乎无alpha，建议删除")
        elif rank_ic > 0.03:
            print("    结论: 正向alpha，保留")
        elif rank_ic < -0.03:
            print("    结论: 负向因子，反转后可用")
        else:
            print("    结论: 信号较弱，观察")
    return ic, rank_ic


def analyze_all_factors(df, factor_cols, forward_return_col='fwd_ret'):
    results = []
    for col in factor_cols:
        ic      = calc_ic(df, col, forward_return_col)
        rank_ic = calc_rank_ic(df, col, forward_return_col)
        results.append({'factor': col, 'ic': ic, 'rank_ic': rank_ic})
    res_df = pd.DataFrame(results).sort_values('rank_ic', ascending=False)
    print("\n因子IC排名：")
    print(res_df.to_string(index=False))
    return res_df


# ===============================
# 4. 数据准备
# ===============================

def get_forward_return(pro, ts_codes, date, n_days):
    """取 date 之后 n_days 个交易日的收益率"""
    end = datetime.strptime(date, '%Y%m%d')
    future_end = (end + timedelta(days=n_days * 3)).strftime('%Y%m%d')
    cal = get_trade_cal(pro, date, future_end)
    cal = cal.sort_values('cal_date').reset_index(drop=True)
    cal = cal[cal['cal_date'] > date]
    if len(cal) < n_days:
        return {}
    target = cal.iloc[n_days - 1]['cal_date']

    p0 = get_price_on_date(pro, date);   time.sleep(1)
    p1 = get_price_on_date(pro, target); time.sleep(1)

    p0 = p0.set_index('ts_code')['close']
    p1 = p1.set_index('ts_code')['close']

    returns = {}
    for ts in ts_codes:
        if ts in p0.index and ts in p1.index:
            v0, v1 = p0[ts], p1[ts]
            if pd.notna(v0) and pd.notna(v1) and v0 > 0:
                returns[ts] = (v1 / v0 - 1) * 100
    return returns


def build_factor_dataset(pro, industry_map, financials_all):
    """
    遍历截面日，拉因子值 + 未来收益，存成一个大 DataFrame
    这个数据集就是IC分析的输入
    """
    cal = get_trade_cal(pro, ANALYSIS_START, ANALYSIS_END)
    cal = cal.sort_values('cal_date').reset_index(drop=True)
    sample_dates = cal['cal_date'].tolist()[::SAMPLE_FREQ]
    print(f"共{len(sample_dates)}个截面日\n")

    all_records = []

    for i, date in enumerate(sample_dates):
        print(f"[{i+1}/{len(sample_dates)}] {date}", end='  ')
        try:
            basic       = get_daily_basic(pro, date);                  time.sleep(1)
            mf          = get_moneyflow(pro, date);                    time.sleep(1)
            margin      = get_margin(pro, date);                       time.sleep(1)
            momentum_df = calculate_momentum_fast(pro, date, days=60); time.sleep(1)
            market_state, _ = get_market_state(pro, date)
            financials_df   = get_financials_as_of(financials_all, date)

            df_scored = calculate_factors(
                basic, mf, margin, momentum_df,
                market_state, industry_map, financials_df
            )
            ts_codes = df_scored['ts_code'].tolist()

            # 取最常用的 20 日未来收益
            fwd_rets = get_forward_return(pro, ts_codes, date, 20)

            for _, row in df_scored.iterrows():
                ts = row['ts_code']
                if ts not in fwd_rets:
                    continue
                rec = {'date': date, 'ts_code': ts, 'fwd_ret': fwd_rets[ts]}
                for col in FACTOR_COLS:
                    rec[col] = row.get(col, np.nan)
                all_records.append(rec)

            print(f"✓ {len(fwd_rets)}只")

        except Exception as e:
            print(f"跳过({e})")
            continue

        time.sleep(2)

    df = pd.DataFrame(all_records)
    df.to_csv('factor_dataset.csv', index=False)
    print(f"\n数据集已保存：factor_dataset.csv（{len(df)}条）")
    return df


# ===============================
# 5. 主流程
# ===============================

def run():
    pro = init()
    print("=" * 55)
    print("AlphaDog 因子IC分析")
    print("=" * 55)

    print("\n加载基础数据...")
    industry_map   = get_industry_map(pro);      time.sleep(2)
    financials_all = get_cached_financials(pro); time.sleep(2)

    # 如果已有数据集直接加载，省时间
    try:
        df = pd.read_csv('factor_dataset.csv')
        print(f"读取已有数据集：{len(df)}条记录")
    except FileNotFoundError:
        print("\n构建因子数据集（首次运行约需1小时）...")
        df = build_factor_dataset(pro, industry_map, financials_all)

    if len(df) == 0:
        print("数据集为空，退出")
        return

    # ── 批量IC分析 ──
    print("\n" + "=" * 55)
    print("全量IC分析（20日预测）")
    res = analyze_all_factors(df, FACTOR_COLS)

    # ── 逐因子详细分析 + 稳定性 ──
    print("\n" + "=" * 55)
    print("逐因子详细分析")
    for col in FACTOR_COLS:
        analyze_factor(df, col)
        ic_series = rolling_ic(df, col, window=5)
        print_ic_stability(ic_series, col)

    # ── 保存结果 ──
    res.to_csv('factor_ic_result.csv', index=False)
    print(f"\n结果已保存：factor_ic_result.csv")
    return res


if __name__ == "__main__":
    run()