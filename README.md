# AlphaDog 量化交易系统

## 一句话描述

个人量化+AI Agent的A股交易系统，目标年化超额>15%，最大回撤<20%，夏普>1.5

## 快速启动

```bash
cd ~/alphadog
source venv/bin/activate
python -m model.factor       # 今日选股
python -m backtest.engine    # 跑回测
```

## 项目结构

alphadog/

├── data/

│ ├── fetcher.py # 所有数据接口（Tushare）

│ └── cache/

│ ├── financials.csv # 财务数据缓存（季度自动更新）

│ └── financials_meta.txt # 缓存元数据

├── model/

│ └── factor.py # 因子模型：选股+打分

├── backtest/

│ └── engine.py # 回测框架

├── agent/ # 待开发：新闻/财报/政策信号

├── risk/ # 待开发：止损/仓位管理

├── execution/ # 待开发：下单执行

└── README.md

## 当前因子

| 因子           | 类型 | 打分方式   |
| -------------- | ---- | ---------- |
| PE             | 估值 | 行业内排名 |
| PB             | 估值 | 行业内排名 |
| 动量(60天)     | 技术 | 行业内排名 |
| 主力资金净流入 | 情绪 | 全市场排名 |
| ROE            | 质量 | 行业内排名 |
| 融资买入额     | 情绪 | 全市场排名 |

## 动态权重

| 市场状态 | PE  | PB  | 动量 | 资金流 | ROE | 融资 |
| -------- | --- | --- | ---- | ------ | --- | ---- |
| 牛市     | 10% | 10% | 30%  | 25%    | 15% | 10%  |
| 熊市     | 30% | 25% | 10%  | 10%    | 20% | 5%   |
| 震荡     | 20% | 15% | 20%  | 20%    | 15% | 10%  |

## 架构演进计划

现在（串行） 目标（并行）

data data

↓ /

model 因子模型 Signal Engine

↓ \ /

execution Fusion

↓

execution

## 进度

- [x] Day 1：环境搭建，Tushare接入，factor第一版
- [x] Day 2：行业中性化，动态权重，智能缓存，回测框架
- [ ] Day 3：跑回测，分析结果
- [ ] Day 4：walk-forward验证
- [ ] Day 5+：LightGBM，Agent层，Fusion层，风控，执行层

## 关键设计决定

- Agent不直接买卖，只输出信号（情绪分/风险分），由Fusion层合并决策
- features和model后续拆开，加LightGBM时重构
- 行业中性化：行业内排名打分，不硬限制每行业几只
- NaN因子自动跳过，权重动态重新分配给有效因子
- 财务缓存：7天内读缓存，超7天检测是否有新财报

## 数据源

- Tushare Pro 2000积分（已充值）
- DeepSeek API（Agent待接入）

## 环境

- Python 3.13，venv路径：~/alphadog/venv
- 依赖：tushare, pandas, numpy
