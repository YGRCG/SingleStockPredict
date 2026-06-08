# SingleStockPredict

多周期单只股票未来三日涨跌预测。覆盖数据下载、特征工程、标签构建、滚动训练、回测全流程。

## 环境安装

```powershell
# 安装 uv（如未安装）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 安装依赖
uv sync

# 可选：PyTorch（LSTM 模型）
uv sync --extra torch

# 可选：开发依赖（pytest、notebook）
uv sync --extra dev
```

## 快速开始

**1. 修改配置**

编辑 `config/config.yaml`，设置股票代码和日期范围：

```yaml
stock:
  symbol: "000001.SZ"   # 上证用 sh.xxxxxx，深证用 sz.xxxxxx

data:
  start_date: "2015-01-01"
  end_date:   "2025-12-31"
```

**2. 一键运行完整流水线**

```powershell
uv run python scripts/run_pipeline.py
```

**3. 分步运行**

```powershell
# 单独训练（--model 可选 lgbm / xgb / lstm）
uv run python scripts/run_train.py --model lgbm

# 加载已有模型回测
uv run python scripts/run_backtest.py --model lgbm
```

**4. 运行测试**

```powershell
uv run pytest
```

## 项目结构

```
config/          配置文件（数据、特征、标签、模型超参）
src/
  data/          baostock 数据下载与加载
  features/      技术指标、多周期对齐、形态特征
  labels/        标签构建（binary / ternary / triple_barrier）
  models/        LightGBM / XGBoost / LSTM
  training/      滚动训练、时序交叉验证
  backtest/      回测引擎、绩效指标
scripts/         命令行入口
output/          模型、预测结果、报告图表
```

## 标签类型

| 类型 | 说明 |
|------|------|
| `binary` | 未来 N 日涨 → 1，跌 → 0 |
| `ternary` | 涨幅 > threshold → 2，跌幅 > threshold → 0，震荡 → 1 |
| `return` | 未来 N 日对数收益（回归） |
| `triple_barrier` | 三重障碍法：止盈先触及 → 1，止损先触及 → -1，超时 → 符号 |
