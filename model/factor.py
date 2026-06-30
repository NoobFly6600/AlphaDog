import pandas as pd
import numpy as np
import sys
import time
sys.path.append('.')
from data.fetcher import (
    init, get_daily_basic, get_moneyflow,
    get_margin, get_price_on_date,
    get_nth_trading_day_before, get_industry_map,
    get_cached_financials, get_trade_cal
)


def get_market_state(pro, trade_date):
    """判断市场状态：牛市/熊市/震荡"""
    from datetime import datetime, timedelta
    end   = datetime.strptime(trade_date, '%Y%m%d')
    start = (end - timedelta(days=45)).strftime('%Y%m%d')

    idx = pro.index_daily(
        ts_code='399300.SZ',
        start_date=start,
        end_date=trade_date,
        fields='trade_date,close'
    )
    idx = idx.sort_values('trade_date')

    if len(idx) < 20:
        return 'neutral', 50

    ma20    = idx['close'].tail(20).mean()
    current = idx['close'].iloc[-1]
    pct_vs_ma = (current - ma20) / ma20 * 100

    if pct_vs_ma > 2:
        return 'bull', 70
    elif pct_vs_ma < -2:
        return 'bear', 30
    else:
        return 'neutral', 50


def get_dynamic_weights(market_state):
    if market_state == 'bull':
        return {
            'pe':        0.10,
            'pb':        0.10,
            'momentum':  0.30,
            'moneyflow': 0.25,
            'margin':    0.15,
        }
    elif market_state == 'bear':
        return {
            'pe':        0.30,
            'pb':        0.25,
            'momentum':  0.10,
            'moneyflow': 0.10,
            'margin':    0.25,
        }
    else:
        return {
            'pe':        0.20,
            'pb':        0.15,
            'momentum':  0.20,
            'moneyflow': 0.20,
            'margin':    0.25,
        }


def calculate_momentum_fast(pro, trade_date, days=60):
    """快速计算全市场动量，两次API call搞定"""
    price_now = get_price_on_date(pro, trade_date)
    time.sleep(3)

    day_n = get_nth_trading_day_before(pro, trade_date, days)
    if day_n is None:
        return pd.DataFrame(columns=['ts_code', 'momentum'])
    time.sleep(3)

    price_then = get_price_on_date(pro, day_n)
    merged = price_now[['ts_code', 'close']].merge(
        price_then[['ts_code', 'close']],
        on='ts_code', suffixes=('_now', '_then')
    )
    merged['momentum'] = (merged['close_now'] / merged['close_then'] - 1) * 100
    return merged[['ts_code', 'momentum']]


def score_global(series, ascending=True):
    """全市场排名打分，NaN保持NaN"""
    ranked = series.rank(pct=True, na_option='keep') * 100
    return (100 - ranked) if ascending else ranked


def score_within_industry(df, factor_col, ascending=True):
    """行业内排名打分，NaN保持NaN"""
    def rank_group(group):
        ranked = group.rank(pct=True, na_option='keep') * 100
        return (100 - ranked) if ascending else ranked
    return df.groupby('industry')[factor_col].transform(rank_group)


def weighted_score(row, w):
    """NaN因子自动跳过，权重重新分配给有效因子"""
    factors = {
        'pe':        ('pe_score',        w['pe']),
        'pb':        ('pb_score',        w['pb']),
        'momentum':  ('momentum_score',  w['momentum']),
        'moneyflow': ('moneyflow_score', w['moneyflow']),
        'margin':    ('margin_score',    w['margin']),
    }
    total_weight = 0
    total_score  = 0
    for _, (col, wt) in factors.items():
        val = row[col] if col in row.index else np.nan
        if pd.notna(val):
            total_score  += val * wt
            total_weight += wt
    return total_score / total_weight if total_weight > 0 else np.nan


def calculate_factors(daily_basic, moneyflow, margin, momentum_df,
                      market_state, industry_map, financials_df=None):
    df = daily_basic.copy()

    # 合并行业
    df = df.merge(industry_map, on='ts_code', how='left')
    df['industry'] = df['industry'].fillna('其他')

    # 合并资金流向
    df = df.merge(
        moneyflow[['ts_code', 'net_mf_amount', 'buy_elg_amount', 'sell_elg_amount']],
        on='ts_code', how='left'
    )

    # 合并融资融券
    margin_cols = ['ts_code'] + [c for c in ['rzye', 'rzmre'] if c in margin.columns]
    df = df.merge(margin[margin_cols], on='ts_code', how='left')
    if 'rzmre' not in df.columns:
        df['rzmre'] = 0.0
    df['rzmre'] = df['rzmre'].fillna(0.0)

    # 合并动量
    if momentum_df is not None and len(momentum_df) > 0:
        df = df.merge(momentum_df[['ts_code', 'momentum']], on='ts_code', how='left')
    else:
        df['momentum'] = np.nan

    # 合并财务数据（point-in-time，由engine传入）
    if financials_df is not None and len(financials_df) > 0:
        df = df.merge(
            financials_df[['ts_code', 'netprofit_yoy', 'grossprofit_margin']],
            on='ts_code', how='left'
        )
    else:
        df['netprofit_yoy']      = np.nan
        df['grossprofit_margin'] = np.nan

    # 基础过滤
    df = df[df['pe'] > 0]
    df = df[df['pe'] < 100]
    df = df[df['circ_mv'] > 500000]          # 流通市值从2亿提到10亿
    df = df[~df['ts_code'].str.endswith('.BJ')]  # 排除北交所
    df = df.dropna(subset=['pe', 'pb'])
    df = df.reset_index(drop=True)

    # 因子打分
    # IC分析结论：
    # pb        → 正向，保留（最强，IC_IR=3.2）
    # moneyflow → 正向，保留（IC_IR=1.1）
    # pe        → 正向弱，保留（IC_IR=0.98）
    # momentum  → 反转！ascending=True（A股反转效应）
    # margin    → 反转！ascending=True（融资追高是反向信号）
    # roe       → 删除（IC_IR=-0.47，无效）

    df['pe_score']        = score_within_industry(df, 'pe',          ascending=True)
    df['pb_score']        = score_within_industry(df, 'pb',          ascending=True)
    df['moneyflow_score'] = score_global(df['net_mf_amount'],        ascending=False)
    df['momentum_score']  = score_within_industry(df, 'momentum', ascending=False)  # 改回False
    df['margin_score']    = score_global(df['rzmre'],             ascending=False)   # 改回False

    # 综合打分（ROE已删除）
    w = get_dynamic_weights(market_state)
    df['total_score'] = df.apply(lambda row: weighted_score(row, w), axis=1)

    return df


def select_stocks(df, top_n=20):
    """选出综合得分最高的N只股票"""
    selected = df.nlargest(top_n, 'total_score')
    cols = ['ts_code', 'industry', 'pe', 'pb', 'turnover_rate',
            'net_mf_amount', 'momentum', 'total_score']
    existing = [c for c in cols if c in selected.columns]
    return selected[existing].reset_index(drop=True)


if __name__ == "__main__":
    pro = init()
    trade_date = "20241231"

    print("[1/6] 判断市场状态...")
    market_state, sentiment = get_market_state(pro, trade_date)
    print(f"  状态：{market_state}，情绪：{sentiment}")
    print(f"  权重：{get_dynamic_weights(market_state)}")
    time.sleep(3)

    print("\n[2/5] 拉取估值数据...")
    basic = get_daily_basic(pro, trade_date)
    print(f"  {len(basic)}只股票")
    time.sleep(3)

    print("\n[3/5] 拉取行业信息...")
    industry_map = get_industry_map(pro)
    print(f"  {len(industry_map)}只股票")
    time.sleep(3)

    print("\n[4/5] 拉取资金流向...")
    mf = get_moneyflow(pro, trade_date)
    print(f"  {len(mf)}只股票")
    time.sleep(3)

    print("\n[5/5] 计算全市场动量...")
    momentum_df = calculate_momentum_fast(pro, trade_date, days=60)
    print(f"  {len(momentum_df)}只股票")

    print("\n计算综合因子...")
    df_scored = calculate_factors(
        basic, mf, None, momentum_df,
        market_state, industry_map
    )
    result = select_stocks(df_scored, top_n=20)

    print(f"\n{'='*50}")
    print(f"AlphaDog选股结果（{trade_date}，市场:{market_state}）：")
    print(result.to_string(index=False))
    print(f"\n行业分布：")
    print(result['industry'].value_counts())