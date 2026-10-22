# 第 1、2 部分的数据约定

输入根目录包含 `Dataset/`、`Dataset (With Summer GT)/` 和 `FANK_Sat.png`。
解析文件名 `RxxNyy_Season_Modality-suffix.png` 或 `RxxUyy_...`；`Season` 为
Summer/Winter，`Modality` 为 RGB/D/GT。R04 后缀可能保留旧文件名，身份只取前缀。

RGB：可读取 PNG，通常 640×480；审计保留实际尺寸和 SHA-256。
D：与 RGB 同尺寸的 uint8 单通道，或三个完全相同的灰度通道；其他编码标记不可用。
灰度范围 0–255，物理单位未知。零值比例是统计量，不能直接换算距离。
GT：二维掩膜，记录实际尺寸和标签值；不用于同树身份评分或本轮划分。

缺失/损坏 RGB 保留在审计表，相应 N 配对记录排除原因；缺失/异常 D 保留有效 RGB 配对。
同身份、季节、模态存在多个文件时报错，避免任意挑选。U 不参与 N 身份配对和划分。

输出：

| 文件 | 含义 |
|---|---|
| observation_index.csv | 每个子集/编号/季节一条观测，包含可读性、尺寸、哈希和 D/GT 状态 |
| config/pair_review.csv | 全部 N 候选配对及人工核对状态；required_review 指示要求抽查的集合 |
| split_manifest.csv | 每个有效 N 身份的 Summer/Winter 路径及行级 split |
| *_queries.csv / *_gallery.csv | 每部分的夏季查询和冬季图库；编号为未来监督/评估信息 |
| leakage_checks.csv | 三组集合两两之间的行、身份、路径、内容哈希交集检查 |
| quality_issues.csv | 缺失、不可读或不支持的观测 |
| special_filenames.csv | R04 等包含额外字母后缀的观测 |
| duplicate_rgb_content.csv | 内容相同的 RGB，包括跨 N/U 子集重叠 |
| excluded_n_pairs.csv | 自动质量排除和人工 rejected 的原因 |
| summary.json | 配对数量、核对状态、行级划分、版本及清单哈希 |
| review_sheets/ | 每个 required 配对的并排原始 RGB 图 |

路径相对项目根目录保存，随整个项目迁移。编号索引图的横轴 R、纵轴 N 不具有米制意义。
人工记录只接受 pending/confirmed/rejected；程序不把自动检查当作人工确认。
