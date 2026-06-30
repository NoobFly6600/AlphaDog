import pandas as pd
import numpy as np
import sys
import time
sys.path.append('.')
from data.fetcher import (
    init, get_daily_basic, get_moneyflow,
    get_margin, get_price_on_date,
    get_nth_trading_day_before, get_industry_map,
    get_cached_financials, get_trade_cal,
    get_financials_as_of
)
from model.factor import (
    get_market_state, calculate_momentum_fast,
    calculate_factors, select_stocks
)
from datetime import datetime, timedelta

# ── 风控参数 ──────────────────────────────────────────────
BASE_POSITION  = {'bull': 1.0, 'neutral': 1.0, 'bear': 1.0}
STOPLOSS_THR   = -999
STOPLOSS_MULTI = 0.5
TRADE_COST     = 0.20


def get_nearest_trading_day(pro, date_str):
    end   = date_str
    start = (datetime.strptime(date_str, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d')
    cal   = get_trade_cal(pro, start, end)
    cal   = cal.sort_values('cal_date').reset_index(drop=True)
    return cal.iloc[-1]['cal_date'] if len(cal) > 0 else date_str


def get_rebalance_dates(pro, start_date, end_date, frequency=14):
    cal = get_trade_cal(pro, start_date, end_date)
    cal = cal.sort_values('cal_date').reset_index(drop=True)
    return cal['cal_date'].tolist()[::frequency]


def get_period_return(pro, ts_code_list, start_date, end_date):
    returns = {}
    price_start = get_price_on_date(pro, start_date)
    time.sleep(2)
    price_end = get_price_on_date(pro, end_date)

    ps = price_start.set_index('ts_code')['close']
    pe = price_end.set_index('ts_code')['close']

    skipped = 0
    for ts_code in ts_code_list:
        if ts_code in ps.index and ts_code in pe.index:
            p0, p1 = ps[ts_code], pe[ts_code]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                skipped += 1
                continue
            returns[ts_code] = (p1 / p0 - 1) * 100
        else:
            skipped += 1
    if skipped:
        print(f"  跳过{skipped}只停牌/退市")
    return returns


def get_position_ratio(market_state, last_net_return):
    ratio   = BASE_POSITION.get(market_state, 0.8)
    reasons = [f"市场{market_state}→基础仓位{int(ratio*100)}%"]
    if last_net_return is not None and last_net_return < STOPLOSS_THR:
        ratio  *= STOPLOSS_MULTI
        reasons.append(f"上期亏{abs(last_net_return):.1f}%触发止损→仓位降至{int(ratio*100)}%")
    return ratio, reasons


def calculate_sharpe(returns_list, risk_free_annual=0.02, periods_per_year=18):
    if len(returns_list) < 2:
        return np.nan
    r  = np.array(returns_list) / 100
    rf = risk_free_annual / periods_per_year
    ex = r - rf
    return (ex.mean() / ex.std()) * np.sqrt(periods_per_year) if ex.std() > 0 else np.nan


def run_backtest(
    start_date='20230101',
    end_date='20241231',
    top_n=20,
    frequency=14,
    initial_capital=10000
):
    pro = init()

    print(f"AlphaDog回测开始")
    print(f"  区间：{start_date} → {end_date}")
    print(f"  换仓频率：每{frequency}个交易日")
    print(f"  选股数量：{top_n}只")
    print(f"  初始资金：{initial_capital}元")
    print(f"  熊市仓位：{int(BASE_POSITION['bear']*100)}%  止损线：{STOPLOSS_THR}%")
    print(f"  财报：point-in-time（按公告日过滤）")
    print("=" * 55)

    rebalance_dates = get_rebalance_dates(pro, start_date, end_date, frequency)
    print(f"\n共{len(rebalance_dates)}个换仓周期")

    print("\n预加载数据...")
    industry_map   = get_industry_map(pro);      time.sleep(3)
    financials_all = get_cached_financials(pro); time.sleep(3)

    portfolio_value    = initial_capital
    portfolio_history  = []
    period_returns_log = []
    holdings           = []
    last_net_return    = None
    market_state       = 'neutral'

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]
        print(f"\n[{i+1}/{len(rebalance_dates)-1}] 换仓日：{date} → 持有至：{next_date}")

        try:
            # 风控：决定本期仓位
            ratio, reasons = get_position_ratio(market_state, last_net_return)
            for r in reasons:
                print(f"  {r}")

            # 拉数据
            basic       = get_daily_basic(pro, date);                  time.sleep(2)
            mf          = get_moneyflow(pro, date);                    time.sleep(2)
            margin      = get_margin(pro, date);                       time.sleep(2)
            momentum_df = calculate_momentum_fast(pro, date, days=60); time.sleep(2)
            market_state, _ = get_market_state(pro, date)

            # point-in-time 财报
            financials_df = get_financials_as_of(financials_all, date)

            # 选股
            df_scored    = calculate_factors(
                basic, mf, margin, momentum_df,
                market_state, industry_map, financials_df
            )
            selected     = select_stocks(df_scored, top_n=top_n)
            new_holdings = selected['ts_code'].tolist()

            # 计算上期收益
            if holdings:
                period_rets = get_period_return(pro, holdings, date, next_date)
                time.sleep(2)

                if period_rets:
                    gross = np.mean(list(period_rets.values()))
                    net   = gross - TRADE_COST
                    last_net_return = net

                    invested        = portfolio_value * ratio
                    cash_kept       = portfolio_value - invested
                    invested_after  = invested * (1 + net / 100)
                    portfolio_value = invested_after + cash_kept

                    period_returns_log.append((gross, net, ratio))

                    print(f"  仓位{int(ratio*100)}%  持仓{len(period_rets)}/{len(holdings)}只")
                    print(f"  毛收益{gross:.2f}%  净收益{net:.2f}%  组合{portfolio_value:.0f}元")

            holdings = new_holdings
            portfolio_history.append({
                'date':            date,
                'portfolio_value': portfolio_value,
                'market_state':    market_state,
                'position_ratio':  ratio,
                'holdings':        ','.join(new_holdings),
            })

        except Exception as e:
            print(f"  跳过（{e}）")
            continue

        time.sleep(3)

    # 统计指标
    print("\n" + "=" * 55)
    print("回测结果：")

    history_df = pd.DataFrame(portfolio_history)

    if len(history_df) > 0:
        total_return  = (portfolio_value / initial_capital - 1) * 100
        days          = (datetime.strptime(end_date, '%Y%m%d') -
                         datetime.strptime(start_date, '%Y%m%d')).days
        annual_return = ((portfolio_value / initial_capital) ** (365 / days) - 1) * 100

        history_df['peak']     = history_df['portfolio_value'].cummax()
        history_df['drawdown'] = (history_df['portfolio_value'] - history_df['peak']) / history_df['peak'] * 100
        max_drawdown           = history_df['drawdown'].min()

        net_rets = [x[1] for x in period_returns_log]
        sharpe   = calculate_sharpe(net_rets, periods_per_year=int(252 / frequency))

        actual_start = get_nearest_trading_day(pro, start_date)
        actual_end   = get_nearest_trading_day(pro, end_date)
        idx_s = pro.index_daily(ts_code='399300.SZ', start_date=actual_start,
                                end_date=actual_start, fields='close')
        idx_e = pro.index_daily(ts_code='399300.SZ', start_date=actual_end,
                                end_date=actual_end,   fields='close')
        if len(idx_s) > 0 and len(idx_e) > 0:
            bm_return = (idx_e['close'].values[0] / idx_s['close'].values[0] - 1) * 100
            excess    = total_return - bm_return
        else:
            bm_return = excess = 0

        avg_position = np.mean([x[2] for x in period_returns_log]) * 100 if period_returns_log else 0

        print(f"  总收益率：    {total_return:.2f}%")
        print(f"  年化收益率：  {annual_return:.2f}%")
        print(f"  最大回撤：    {max_drawdown:.2f}%")
        print(f"  夏普比率：    {sharpe:.2f}" if not np.isnan(sharpe) else "  夏普比率：    N/A")
        print(f"  沪深300收益： {bm_return:.2f}%")
        print(f"  超额收益：    {excess:.2f}%")
        print(f"  平均仓位：    {avg_position:.0f}%")
        print(f"  换仓次数：    {len(history_df)}次")

        print(f"\n目标达成情况（年化>15%，回撤<20%，夏普>1.5）：")
        print(f"  年化收益：{'✓' if annual_return > 15  else '✗'} {annual_return:.2f}%")
        print(f"  最大回撤：{'✓' if abs(max_drawdown)<20 else '✗'} {abs(max_drawdown):.2f}%")
        if not np.isnan(sharpe):
            print(f"  夏普比率：{'✓' if sharpe > 1.5 else '✗'} {sharpe:.2f}")

        history_df.to_csv('backtest_result.csv', index=False)
        print(f"\n结果已保存至 backtest_result.csv")

    return history_df


if __name__ == "__main__":
   run_backtest(
    start_date='20200101',
    end_date='20221231',
    top_n=20,
    frequency=14,
    initial_capital=10000
)