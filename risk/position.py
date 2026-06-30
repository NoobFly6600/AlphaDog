"""
risk/position.py
仓位管理：根据市场状态 + 止损信号决定本期仓位比例
"""

# 各市场状态对应的基础仓位
BASE_POSITION = {
    'bull':    1.0,
    'neutral': 0.8,
    'bear':    0.5,
}

# 止损触发阈值（上期净收益低于此值则触发）
STOPLOSS_THRESHOLD = -5.0

# 止损触发后仓位砍半系数
STOPLOSS_FACTOR = 0.5


def get_position_ratio(market_state: str, last_period_return: float | None) -> tuple[float, list[str]]:
    """
    决定本期仓位比例

    参数：
        market_state:       当前市场状态（'bull' / 'neutral' / 'bear'）
        last_period_return: 上期净收益率（百分比，如 -6.5 表示亏了6.5%）
                            第一期传 None

    返回：
        (position_ratio, reasons)
        position_ratio: 0.0~1.0，表示投入资金比例
        reasons:        说明列表，方便打印
    """
    reasons = []

    # 第一步：市场状态决定基础仓位
    base = BASE_POSITION.get(market_state, 0.8)
    reasons.append(f"市场{market_state}→基础仓位{int(base*100)}%")

    # 第二步：检查止损
    ratio = base
    if last_period_return is not None and last_period_return < STOPLOSS_THRESHOLD:
        ratio = ratio * STOPLOSS_FACTOR
        reasons.append(f"上期亏损{last_period_return:.1f}%触发止损→仓位砍半至{int(ratio*100)}%")

    return ratio, reasons


def apply_position(portfolio_value: float, position_ratio: float) -> tuple[float, float]:
    """
    计算实际投入金额和留存现金

    返回：
        (invested, cash)
    """
    invested = portfolio_value * position_ratio
    cash = portfolio_value - invested
    return invested, cash