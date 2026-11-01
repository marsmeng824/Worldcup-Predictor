# World Cup Predictor

一个基于历史国家队比赛数据的世界杯比赛预测 dashboard 原型。

## 功能

- 从历史比赛结果生成训练数据
- 计算球队 tier、Elo、最近 5/10 场进球、失球和净胜球特征
- 训练胜平负分类模型
- 训练进球数回归模型，并根据预测进球推导比赛结果
- 在 Streamlit dashboard 中选择两支球队并查看预测比分和胜平负概率

## 项目结构

```text
app.py                     # Streamlit dashboard
build_analysis_data.py     # 从原始比赛数据生成 analysis_data.csv
train_model.py             # 训练分类模型和进球数回归模型
worldcup_2026_teams.csv    # 目标球队、实力和分组信息
data/international_results.csv
                           # 原始国际比赛结果
```

生成文件不会提交到 GitHub，包括：

```text
models/
data/analysis_data.csv
data/team_current_features.csv
data/model_*.csv
data/regression_*.csv
```

## 运行

```powershell
cd "C:\Users\mengy\Documents\New project\worldcup_predictor"
pip install -r requirements.txt

python build_analysis_data.py
python train_model.py
streamlit run app.py
```

如果想只分析 2014 世界杯之后的比赛：

```powershell
python build_analysis_data.py --analysis-start-date 2014-07-14
python train_model.py
```

## 说明

当前版本是课程/个人项目原型，不是官方预测。模型结果主要依赖历史比赛、人工 tier、Elo 和近期状态特征，不能代表真实比赛结论。
