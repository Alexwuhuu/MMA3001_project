# MMA3001 — Cross-season Apple-tree Matching

当前只实现 **①配置与数据检查、②标签核对与按行划分**。主入口是
[`apple_tree_matching.ipynb`](apple_tree_matching.ipynb)。本轮不进行模型训练。
后续方案为 **Linear Regression baseline + RBF-SVR 主方法**，不使用 HOG。

## 文件结构

```text
MMA3001_project/
├── .github/workflows/ci.yml     # 无需私有数据的测试和合成数据 Notebook 检查
├── .gitignore
├── README.md
├── LICENSE
├── pytest.ini                  # 禁用共享缓存；测试临时文件由 conftest 放入项目 .cache/
├── apple_tree_matching.ipynb    # 主入口，仅第 1、2 部分
├── config/
│   ├── experiment.json         # 默认路径、seed、核对设置
│   ├── requirements.in         # 直接依赖
│   ├── requirements.txt        # 已验证的精确依赖版本，Python 3.12
│   └── pair_review.csv         # 首次运行生成；保留实际人工核对结果
├── doc/                        # 运行约定、数据说明、AI 使用；非 project report
├── res/                        # 静态资源说明
├── src/
│   ├── __init__.py
│   ├── main.py                 # 审计、核对图册、可移植导出
│   └── utils.py                # 文件解析、图像检查、N 配对、按行划分
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # 隔离 Windows 测试临时目录
│   └── test_main.py
├── tools/
│   ├── run_notebook.py         # 用当前 Python 执行 Notebook
│   └── synthetic_data.py       # CI 数据，不能用于果园精度结论
├── data/TreeID/TreeID/          # 用户数据，Git 忽略
└── outputs/data_preparation/   # 生成表格和图册，Git 忽略
```

## 在当前本机运行（PowerShell）

使用已有的 **MMA3001 conda 环境（Python 3.12）**，依赖版本见
`config/requirements.txt`。首次配置或重建环境时执行：

```powershell
cd D:\2026-S2\MMA3001\MMA3001_project
& 'D:\anacoda3\envs\MMA3001\python.exe' -m pip install -r config\requirements.txt
& 'D:\anacoda3\envs\MMA3001\python.exe' -m ipykernel install --sys-prefix --name mma3001 --display-name 'MMA3001 (Python 3.12)'
```

在 VS Code 中打开 `apple_tree_matching.ipynb`，点击右上角内核选择器，选择
**MMA3001**（解释器为 `D:\anacoda3\envs\MMA3001\python.exe`），然后执行
**Restart Kernel and Run All Cells**。第一个代码单元会显示实际使用的 Python 路径。
也可从该环境启动 JupyterLab：

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' -m jupyterlab apple_tree_matching.ipynb
```

不打开浏览器执行 Notebook、运行数据流程或测试：

```powershell
& 'D:\anacoda3\envs\MMA3001\python.exe' tools\run_notebook.py --in-place
& 'D:\anacoda3\envs\MMA3001\python.exe' -m src.main
& 'D:\anacoda3\envs\MMA3001\python.exe' -m pytest -q
& 'D:\anacoda3\envs\MMA3001\python.exe' tools\run_notebook.py --synthetic
```

项目已有的 `.venv` 也可以运行；将上述解释器换成 `.\.venv\Scripts\python.exe`。
`nbformat`、`nbclient` 是自动执行工具的依赖，交互式数据流程不依赖它们。
版本展示和结果汇总在缺少版本元数据时记录为空，不中断计算。

其他机器使用 Python 3.12 创建自己的环境，再安装 `config/requirements.txt`。
原始数据需单独放置；可创建被 Git 忽略的 `config/local.json`：

```json
{"data_root": "D:/datasets/TreeID/TreeID"}
```

## 编号与核对

- R 是果园行号；N 是可建立跨季节配对的树编号。
- U 是未匹配观测，不生成同编号正样本或未经核实的负样本。
- N 配对先依文件名前缀建立，再抽查每行两对及 R04 全部配对。
- `config/pair_review.csv` 中 `pending` 表示尚未人工核实；检查图册后填写
  `confirmed` 或 `rejected`，可在 `human_notes` 中记录依据。再次运行保留已有决定。
- 20 行按 seed=42 分成 12/4/4 行，禁止身份、行、路径或相同 RGB 内容跨集合。
- 人工检查未完成前，输出是基于文件名标签的**暂定数据准备结果**。

详细输入输出见 [`doc/data_contract.md`](doc/data_contract.md)，AI 使用记录见
[`doc/ai_usage.md`](doc/ai_usage.md)。原图和生成图册不自动加入 Git。

## 数据来源与边界

参考 [Chen et al., Semantic Segmentation for Partially Occluded Apple Trees Based on Deep Learning](https://arxiv.org/pdf/2010.06879)。
论文中的 521 张分割图像与当前本地 N 子集数量不同，本项目独立划分数据。
8-bit D 导出图的物理编码尚不明确，当前不估计真实三维坐标或米制地图。
代码许可证见 `LICENSE`，不代表原始数据或论文的许可。

本任务按要求只保留工作区修改，不执行 commit 或 push。
