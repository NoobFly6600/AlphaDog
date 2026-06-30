import tushare as ts
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta

TOKEN = "51f5b63f964eebab612702cbbe36426a8b4114bbf27d2ed11c7fa1c4"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

def init():
    ts.set_token(TOKEN)
    return ts.pro_api()

def get_stock_list(pro):
    return pro.stock_basic(
        exchange='', list_status='L',
        fields='ts_code,symbol,name,industry,list_date'
    )

def get_industry_map(pro):
    return pro.stock_basic(
        exchange='', list_status='L',
        fields='ts_code,industry'
    )

def get_daily_basic(pro, trade_date):
    return pro.daily_basic(
        trade_date=trade_date,
        fields='ts_code,trade_date,pe,pb,ps,dv_ratio,turnover_rate,circ_mv,volume_ratio'
    )

def get_moneyflow(pro, trade_date):
    return pro.moneyflow(
        trade_date=trade_date,
        fields='ts_code,trade_date,buy_lg_amount,sell_lg_amount,net_mf_amount,buy_elg_amount,sell_elg_amount'
    )

def get_financials(pro, ts_code):
    return pro.fina_indicator(
        ts_code=ts_code,
        fields='ts_code,ann_date,end_date,roe,grossprofit_margin,debt_to_assets,netprofit_yoy,or_yoy,current_ratio'
    )

def get_index_daily(pro, trade_date):
    return pro.index_daily(
        ts_code='399300.SZ',
        start_date=trade_date,
        end_date=trade_date
    )

def get_margin(pro, trade_date):
    return pro.margin_detail(
        trade_date=trade_date,
        fields='ts_code,trade_date,rzye,rzmre,rqye'
    )

def get_price_on_date(pro, trade_date):
    return pro.daily(
        trade_date=trade_date,
        fields='ts_code,trade_date,close,pct_chg'
    )

def get_trade_cal(pro, start_date, end_date):
    return pro.trade_cal(
        exchange='SSE',
        start_date=start_date,
        end_date=end_date,
        is_open='1',
        fields='cal_date'
    )

def get_nth_trading_day_before(pro, trade_date, n):
    end = datetime.strptime(trade_date, '%Y%m%d')
    start = (end - timedelta(days=n * 2)).strftime('%Y%m%d')
    cal = get_trade_cal(pro, start, trade_date)
    cal = cal.sort_values('cal_date').reset_index(drop=True)
    cal = cal[cal['cal_date'] < trade_date]
    if len(cal) < n:
        return None
    return cal.iloc[-n]['cal_date']

def _get_latest_report_date_online(pro):
    try:
        df = pro.fina_indicator(ts_code='000001.SZ', fields='ts_code,ann_date')
        return df['ann_date'].max()
    except:
        return None

def _read_cache_meta(meta_path):
    try:
        with open(meta_path, 'r') as f:
            lines = f.read().strip().split('\n')
        cache_date  = lines[0] if len(lines) > 0 else None
        report_date = lines[1] if len(lines) > 1 else None
        return cache_date, report_date
    except:
        return None, None

def _write_cache_meta(meta_path, report_date):
    with open(meta_path, 'w') as f:
        f.write(datetime.now().strftime('%Y%m%d') + '\n')
        f.write(str(report_date) + '\n')

def should_update_financials(pro, cache_path, meta_path):
    cache_date, cached_report_date = _read_cache_meta(meta_path)
    if cache_date is None or cached_report_date is None:
        return True
    days_old = (datetime.now() - datetime.strptime(cache_date, '%Y%m%d')).days
    if days_old < 7:
        print(f"  缓存{days_old}天前建立，无需检查")
        return False
    print(f"  缓存{days_old}天前建立，检查是否有新财报...")
    latest_online = _get_latest_report_date_online(pro)
    if latest_online is None:
        return False
    if latest_online > cached_report_date:
        print(f"  发现新财报（{cached_report_date} → {latest_online}），需要更新")
        return True
    else:
        print(f"  财报无更新（最新：{latest_online}），继续用缓存")
        return False

def get_cached_financials(pro, force_update=False):
    """
    获取全市场完整财报历史（所有期次，含ann_date）
    用于回测时按公告日过滤，避免未来函数

    关键改动：
    - 之前只存每只股票最新一期
    - 现在存所有历史记录，包含 ann_date
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, 'financials.csv')
    meta_path  = os.path.join(CACHE_DIR, 'financials_meta.txt')

    if not force_update and os.path.exists(cache_path) and os.path.exists(meta_path):
        if not should_update_financials(pro, cache_path, meta_path):
            print(f"  读取财务缓存...")
            return pd.read_csv(cache_path, dtype={'ann_date': str, 'end_date': str})

    print("  拉取全市场完整财报历史（含所有期次，约需30-40分钟）...")
    stocks = get_stock_list(pro)
    ts_code_list = stocks['ts_code'].tolist()
    total = len(ts_code_list)

    results = []
    for i, ts_code in enumerate(ts_code_list):
        try:
            df = get_financials(pro, ts_code)
            if df is not None and len(df) > 0:
                # 保留所有历史期次，不再只取 latest
                df = df[['ts_code', 'ann_date', 'end_date',
                          'roe', 'netprofit_yoy', 'grossprofit_margin',
                          'debt_to_assets']].copy()
                df = df.dropna(subset=['ann_date'])
                df['ann_date'] = df['ann_date'].astype(str)
                df['end_date'] = df['end_date'].astype(str)
                results.append(df)
        except:
            pass
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{total}")
        time.sleep(0.3)

    fin_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    fin_df.to_csv(cache_path, index=False)

    latest_ann = fin_df['ann_date'].max() if 'ann_date' in fin_df.columns else '00000000'
    _write_cache_meta(meta_path, latest_ann)

    print(f"  完成，{len(fin_df)}条财报记录已缓存，最新公告日：{latest_ann}")
    return fin_df


def get_financials_as_of(financials_all, as_of_date):
    """
    核心函数：按公告日过滤，返回每只股票在 as_of_date 当天能看到的最新财报
    解决未来函数问题的关键

    参数：
        financials_all: get_cached_financials() 返回的完整历史DataFrame
        as_of_date:     回测当天日期（字符串，如 '20230601'）

    返回：
        DataFrame，每只股票一行，只包含当天已公告的最新数据
    """
    # 只保留已公告的记录
    visible = financials_all[financials_all['ann_date'] <= as_of_date].copy()

    if len(visible) == 0:
        return pd.DataFrame()

    # 每只股票取公告日最新的那条
    visible = visible.sort_values('ann_date')
    latest = visible.groupby('ts_code').last().reset_index()

    return latest[['ts_code', 'roe', 'netprofit_yoy', 'grossprofit_margin', 'debt_to_assets']]


if __name__ == "__main__":
    pro = init()
    print("测试接口...")

    print("\n1. 行业映射")
    industry = get_industry_map(pro)
    print(f"  {len(industry)}只股票")
    time.sleep(3)

    print("2. 估值数据")
    basic = get_daily_basic(pro, "20241231")
    print(f"  {len(basic)}只股票")
    time.sleep(3)

    print("3. 资金流向")
    mf = get_moneyflow(pro, "20241231")
    print(f"  {len(mf)}只股票")
    time.sleep(3)

    print("4. 融资融券")
    margin = get_margin(pro, "20241231")
    print(f"  {len(margin)}只股票")
    time.sleep(3)

    print("5. 全市场收盘价")
    price = get_price_on_date(pro, "20241231")
    print(f"  {len(price)}只股票")
    time.sleep(3)

    print("6. 财报历史（point-in-time测试）")
    fin_all = get_cached_financials(pro, force_update=True)
    print(f"  总记录数：{len(fin_all)}")
    fin_20230601 = get_financials_as_of(fin_all, '20230601')
    print(f"  2023-06-01可见财报：{len(fin_20230601)}只股票")

    print("\n所有接口测试完成")