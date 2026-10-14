# BIQAE Quantum Risk

**基于 BIQAE 的量子尾部风险估计与 CVaR 投资组合优化实验框架。**

推荐 GitHub 仓库名：`biqae-quantum-risk`。

本项目基于 Python 与 Qiskit，将收益情景建模、损失分布离散化、量子振幅估计和组合优化串联起来，用于比较经典与量子方法在 VaR、ES/CVaR 估计中的精度、区间覆盖率和资源成本，并提供可复现的实验配置与报告。

GitHub 仓库简介（About）可填写：

> Reproducible quantum tail-risk estimation and CVaR portfolio optimization with BIQAE, Qiskit, and classical benchmarks.

建议 Topics：`quantum-computing`、`qiskit`、`amplitude-estimation`、`biqae`、`risk-management`、`cvar`、`portfolio-optimization`。

## 项目定位

- **风险估计**：估计损失分布的风险价值（VaR）及预期损失（ES/CVaR），处理离散分布的分位点原子质量。
- **算法对照**：比较 MC、Sobol、重要性采样、IQAE、MLAE 与 Grid/Normal/Beta-BIQAE。
- **组合优化**：在权重约束下优化 CVaR，并与经典线性规划参考解对照。
- **资源分析**：记录查询预算、shots、Grover 幂、电路资源及实验来源，支持复现和审计。

默认实验使用合成数据和本地模拟后端；开始运行无需量子云账号。真实硬件需要另行接入适配器。查询量对照、模拟结果和电路资源核算应分别解读，不应据此直接推断真实硬件上的运行时间加速。

本项目提供收益分布建模、VaR/ES风险估计、CVaR组合优化，以及经典与量子估计算法对照。代码位于Python包 `q3_biqae`，数据、配置和运行结果保存在项目目录中。

## 实验报告

- [综合实验报告 HTML版](综合实验报告.html)：图片内嵌，可直接打开或单文件转发。
- [综合实验报告 Markdown版](综合实验报告.md)：配套图片位于 `report_assets/`。

报告包含训练基线、风险可靠性、多时间窗口、厚尾与压力分布、分箱及重训、算法版本与参数消融、完整电路资源，以及权重距离、最优性差距和收敛结果。

## 快速开始（Windows PowerShell）

将代码下载或克隆到本地后，进入包含 `pyproject.toml` 的仓库根目录。仓库目录可以命名为 `biqae-quantum-risk`，Python 包名仍为 `q3_biqae`。先用本机 Python 路径安装环境：

```powershell
./scripts/setup.ps1 -Python 'C:/path/to/python.exe'
```

安装完成后，使用项目虚拟环境执行：

```powershell
./.venv-biqae/Scripts/python.exe -m pytest -q
./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/smoke.json --with-optimization
./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/mvp_4asset_16state.json --with-optimization
./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/mvp_4asset_32state.json
```

当前锁文件在 Windows / Python 3.13 环境生成。代码要求 Python ≥3.11；跨平台重新解析依赖后，应重新保存实际版本锁。`environment.lock` 是精确版本列表，不包含可编辑包路径；本包由 setup 脚本单独安装。

## 通用安装方式

主要依赖为 NumPy、SciPy、Qiskit、Qiskit Aer、Qiskit Algorithms、pandas、PyArrow 与 Matplotlib，版本范围见 `pyproject.toml`。

在 Windows PowerShell 中，也可以按版本范围安装开发环境：

```powershell
python -m venv .venv-biqae
./.venv-biqae/Scripts/python.exe -m pip install --upgrade pip
./.venv-biqae/Scripts/python.exe -m pip install -e ".[dev]"
./.venv-biqae/Scripts/python.exe -m q3_biqae validate --config configs/smoke.json
./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/smoke.json --with-optimization
```

Linux / macOS 对应命令：

```bash
python3 -m venv .venv-biqae
.venv-biqae/bin/python -m pip install --upgrade pip
.venv-biqae/bin/python -m pip install -e ".[dev]"
.venv-biqae/bin/python -m q3_biqae validate --config configs/smoke.json
.venv-biqae/bin/python -m q3_biqae run --config configs/smoke.json --with-optimization
```

通用安装按依赖范围解析版本，与锁定环境可能存在差异。建议从小规模 `smoke.json` 开始，再运行 MVP 和参数网格实验。请使用可编辑安装（`-e`），并在仓库根目录执行命令：当前代码按源码位置定位项目内配置、数据与输出目录。

安装后也可使用 `biqae` 命令；使用虚拟环境 Python 的 `-m q3_biqae` 形式无需激活环境。

## 配置与结果

配置采用 JSON 格式，未填写的字段使用 `src/q3_biqae/config.py` 中的默认值。

| 参数 | 含义 |
|---|---|
| `dataset` | 数据来源，例如 `synthetic_student_t` 或 `french` |
| `assets` / `states` / `scenarios` | 资产数、离散状态数与情景样本数 |
| `alpha` | 风险置信水平，例如 `0.95` 或 `0.975` |
| `methods` | 算法列表，例如 `mc`、`sobol`、`is`、`iqae`、`mlae`、`grid_biqae`、`normal_biqae`、`beta_biqae` |
| `backend` | `analytic_shots`、`qiskit_statevector`、`aer_shots` 或 `aer_noise` |
| `budget` / `shots_per_round` | 查询预算与每轮测量次数 |
| `seeds` / `distribution_seed` | 算法随机种子列表与分布生成种子 |
| `optimization_steps` / `w_max` | 优化步数与单个资产权重上限 |
| `output` | 项目内输出路径，默认 `artifacts/runs` |

运行结束后，CLI 打印产物位置。每次实验保存到独立目录 `artifacts/runs/biqae_<UTC>_<id>/`，用于查看结果表、报告、配置快照和资源记录。已有综合报告对应本地历史实验；重新运行的结果应以新生成的产物为准。

## 验证

在 Windows PowerShell 中运行：

```powershell
./.venv-biqae/Scripts/python.exe -m pytest -q
```

Linux / macOS 将 Python 路径替换为 `.venv-biqae/bin/python`。测试覆盖数学边界、量子状态及电路、风险估计、预算、路径约束与集成流程。

## 主要功能

1. 正态、Student-t、t-Copula 非对称边际、状态切换及共损合成分布；统一不可变情景表。
2. 按概率质量进行损失分位分层和尾部加密；保存原始样本、冻结分布、分箱边界及离散化误差。
3. 精确左分位 VaR、含离散原子修正的 RU-ES、CVaR 梯度和 HiGHS 线性规划参考解。
4. MC、独立 scrambled Sobol 重复、重要性采样；记录样本数、IS 似然比与有效样本量。
5. Qiskit 概率状态制备、可逆损失查表、含等号比较器、尾部超额损失受控旋转及反计算验收。
6. 有界 IQAE、MLAE、网格 BIQAE、Normal/Beta 矩匹配 BIQAE；逐轮记录角度、振幅区间、幂、shots 和后验。
7. 不强制判向的 VaR 区间二分、ES 估计及阈值不确定性传播。
8. 联合优化权重和阈值的 RU 次梯度优化；正负收益载荷拆分、封顶单纯形投影、平均权重、样本外风险评价。
9. CLI、参数网格、Parquet/CSV 表、配置/环境/源码/数据哈希、QPY 电路和 Markdown 图表报告。

## 常用命令

```powershell
# 校验配置与输出路径
./.venv-biqae/Scripts/python.exe -m q3_biqae validate --config configs/mvp_4asset_16state.json

# 完整理想电路模拟，有限 shots
./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/statevector_smoke.json

# Aer 去极化噪声模拟
./.venv-biqae/Scripts/python.exe -m q3_biqae run --config configs/hardware_smoke.json

# 优化传导，输出 LP 对照、权重和目标轨迹
./.venv-biqae/Scripts/python.exe -m q3_biqae optimize --config configs/optimization.json

# 30种子参数网格实验
./.venv-biqae/Scripts/python.exe -m q3_biqae sweep --config configs/mvp_4asset_16state.json --grid configs/tail_grid.json

# 命令行交互 Demo
./.venv-biqae/Scripts/python.exe -m q3_biqae demo

# 独立下载 French 原始数据，保存来源和 SHA256
./.venv-biqae/Scripts/python.exe -m q3_biqae download-data

# 导入情景表并记录来源
./.venv-biqae/Scripts/python.exe -m q3_biqae import-frozen 'C:/input/table.npz' --provenance 'source/version/date'
```

French 配置将 `dataset` 设为 `french`，填写项目内 `input_file`、`train_end`、`test_start`。解析第一张日频市值加权行业表，删除缺失标记，百分比除以 100；训练日期严格早于测试起始日期。

## 目录与职责

| 目录 / 文件 | 内容 |
|---|---|
| `configs/` | MVP、尾部网格、优化、理想/噪声 smoke 配置 |
| `src/q3_biqae/config.py` | 参数校验、项目根路径守卫、JSON/哈希工具 |
| `data_*.py`, `scenario_table.py`, `distributions.py` | 数据、来源、Schema、随机生成 |
| `discretization.py`, `exact_reference.py` | 分箱、误差和精确真值 |
| `amplitude.py`, `classical_mc.py`, `qmc.py`, `importance_sampling.py` | 统一载荷估计与经典入口 |
| `state_preparation.py`, `loss_oracle.py`, `tail_oracle.py` | 状态制备、查表、比较、旋转 |
| `measurement.py`, `iqae.py`, `mlae.py`, `biqae.py` | 测量后端、振幅估计、官方 IQAE 参考入口 |
| `var_search.py`, `es_estimation.py` | 风险量估计 |
| `cvar_gradient.py`, `classical_cvar.py`, `optimizer.py` | 有符号梯度、LP、投影优化 |
| `resource_ledger.py`, `statistics.py`, `reporting.py` | 账本、统计、图表报告 |
| `experiment.py`, `cli.py` | 冻结协议与实验调度 |
| `tests/` | 数学边界、状态向量、算法、路径、预算和集成测试 |
| `data/`, `cache/`, `models/`, `tmp/`, `logs/` | 本方案独立输入与工作目录 |
| `artifacts/runs/biqae_<UTC>_<id>/` | 每次运行独立完整产物 |

## 预算与区间定义

- 经典方法预算单位为分布样本；量子方法为展开后的 `A/A†` 调用，每轮成本 `shots × (2k+1)`。并列保存正向 A、逆向 A、shots、Grover 总幂与最大幂。
- 风险预算默认 40% 给 VaR，60% 给 ES；优化预算为每一步全部有符号载荷和阈值载荷的总预算。
- `analytic_shots` 按理想概率进行有限次数随机采样，适合批量算法对照；完整电路资源单独保存在电路审计和资源核算结果中。
- `qiskit_statevector` 执行完整 `Q^k A` 状态向量并按概率采样；`aer_shots` 执行理想 Aer shots；`aer_noise` 在转译后的 u/cx 门添加去极化噪声。
- 核心运行使用索引控制旋转合并查表与载荷的A0电路。显式损失寄存器查表与比较器另有实现及逐位测试。
- MC/IS 的 Hoeffding 区间较保守。Sobol 在重复 t 区间外保留一维网格的差异度误差界，避免重复方差为零时错误输出零宽区间。MLAE profile 区间为渐近区间；贝叶斯可信区间不等于频率覆盖保证。
- 结果保留每个种子的点估计、区间宽度、实测覆盖率、收敛状态和实际查询量。

## 算法版本与参考资料

Normal/Beta-BIQAE采用完整二项似然后验的矩匹配工程近似；网格BIQAE保留完整离散后验。四版本的精度、覆盖率、查询量和后处理时间对照见综合实验报告。

- [BIQAE论文](https://arxiv.org/html/2507.23074v2)
- [Qiskit IQAE API](https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.IterativeAmplitudeEstimation.html)
