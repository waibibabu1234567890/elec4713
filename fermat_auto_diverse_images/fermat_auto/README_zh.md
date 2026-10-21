# 自动选图：127.5°与黄金角成像比较

## 先说明研究范围

48方向周期是数学性质，本实验不预设其为成像差异的原因。比较的是采样数量、探针宽度、图片结构与两种角度之间的交互关系。

**没有有限数据集可以覆盖所有可能的图片或拓扑结构。** 本包自动构建具有明确类别与结构指标的代表性基准，不声称穷尽。默认无需联网、无需下载、无需手动选图；使用程序化生成的基准图片，并非真实摄影样本。自然风格纹理/场景仅是合成图，不能替代独立真实图片的外部验证。

## 自动选择的方式

1. 每类生成8个变体，共160个候选。
2. 对实际保存精度（8位）的振幅图计算三个阈值下的连通分量数、孔洞数、欧拉数、面积比例，以及灰度均值/方差、熵、梯度能量、高频比例、轴向各向异性。
3. 每类先选一个接近该类特征中心的代表，确保20类全部出现。
4. 对特征做log1p变换和尺度标准化，再用最远点采样补足默认24张。该距离是明确的启发式，不是“覆盖所有类型”的证明。
5. 保存候选表、入选表、文件哈希和总览图；相同配置重跑复用同一套图片。

类别包括：

| 类别 | 主要结构 |
|---|---|
| solid | 单一实心连通区域 |
| islands | 多个孤立区域 |
| rings | 单环及多环 |
| perforated | 带多个孔的区域 |
| nested_rings | 嵌套环与周期径向结构 |
| tree | 树状分支 |
| network | 交叉连接与闭环网络 |
| labyrinth | 曲折连通区域与迷宫状结构 |
| fractures | 细线裂隙与折线 |
| stripes | 定向条纹及弯曲条纹 |
| checkerboard | 棋盘状周期结构 |
| radial | 径向辐条 |
| spiral | 螺旋结构 |
| blobs | 平滑局部团块 |
| multiscale | 多尺度随机纹理 |
| gradient | 低频连续梯度 |
| edges | 台阶边缘、窄条与锐边 |
| grains | 不规则晶粒分区 |
| glyphs | 字符式轮廓与孔洞 |
| scene | 合成场景与几何轮廓 |

类别是生成方式标签；阈值下的实际连通/孔洞数以selected_features.csv为准。随机重叠、反色和像素分辨率可能改变拓扑，不保证某类图片总有固定的孔洞数。

拓扑约定：前景4连通，背景8连通；外围补背景，孔洞=背景连通分量数-1；Euler=分量数-孔洞数。阈值0.2/0.5/0.8。它们描述阈值化振幅的数字拓扑，不等于复数物体的完整拓扑，也不是持久同调。

## 安装和运行

Python 3.10+，CPU多进程。解压后在本目录终端执行：

```bash
python -m pip install -r requirements.txt
python test_dataset.py
python test_metrics.py
python run.py --config quick.json --workers 2
```

Windows可双击run_quick.bat，或者用py替代python。

只生成并检查选图，不跑重建：

```bash
python dataset.py --output auto_images --count 24 --size 96
```

查看auto_images/contact_sheet.png和selected_features.csv。

完整实验，默认自动选择24张图：

```bash
python run.py --config main.json --workers 4
```

扩大到40张，仍从默认160个候选选择：

```bash
python run.py --config main.json --auto-count 40 --workers 4
```

修改选图数量或候选数量前，必须修改JSON中的output为新目录。否则程序拒绝混用结果。更大候选池可用--variants 16。auto-count不能小于20（确保每类一张），也不能超过候选总数。

使用自己的真实图片：

```bash
python run.py --config main.json --images images --workers 4
```

此时使用目录内全部支持的PNG/JPG/TIF/BMP/NPY图像，跳过自动生成。读取灰度，缩放到ROI尺寸，不做逐图min-max拉伸。建议提前裁成方形。NPY必须是二维、有限、[0,1]振幅。

## 实验配置

| 配置 | 设置 | 默认重建次数 |
|---|---|---:|
| quick.json | N=96，sigma=6，48像素ROI，20迭代，1种子，24图 | 48 |
| main.json | N=48/96/144/192/240/288/384，sigma=5.25/7/8.75，96像素ROI，1000迭代，3种子 | 3024 |
| matched_updates.json | main条件，改用总位置更新预算96000 | 3024 |
| known_probe.json | main条件，精确已知探针 | 3024 |
| mixed.json | main条件，混合振幅/相位物体 | 3024 |
| phase.json | main条件，纯相位物体 | 3024 |

请先跑quick估计耗时，不建议一次启动全部正式配置。每个配置有独立output；同时启动的命令也不能共用output。

```bash
python run.py --config matched_updates.json --workers 4
python run.py --config known_probe.json --workers 4
```

配置中objects字段由自动选图替换。未知探针是默认；eta_object=0.8、eta_probe=0.1，初始探针宽度是真值的1.1倍。程序是独立ePIE实现，不保证复现原论文数值。

## 公平比较的边界

正式配置的探针计算窗口统一48×48，只改变振幅高斯sigma，避免同时改变检测阵列尺寸；探针单位总强度归一化。强度FWHM=2*sqrt(ln2)*sigma。物体外部固定单位透射，是强先验。

固定扫描半径；r_n=R*sqrt(n/(N-1))。改变N会重新生成径向序列，各N点集不是嵌套的，因此只能解释为不同采样数量的轨迹设计比较，而非单纯增加数据的因果实验。默认固定迭代次数意味着不同N有不同更新次数；matched_updates用于补充对照，向下取整可能余下少量预算，不是严格相同墙钟时间。

整数扫描坐标使用np.rint。重复位置保留。默认无噪声；可设置counts_per_exposure或total_detected_counts，后者大于0时优先。固定检测计数不等于固定入射剂量。

同一图片/采样数/探针/seed下两个角度使用匹配seed，但衍射数据不同，不应理解为相同噪声样本。更新顺序随机性与噪声分别使用独立随机流。

## 输出

- auto_dataset/contact_sheet.png：自动入选图片总览。
- auto_dataset/candidate_features.csv：全部候选结构特征。
- auto_dataset/selected_features.csv、dataset.json：入选类别、特征和哈希。
- results.csv：逐图片/条件/种子的PSNR、SSIM、振幅RMSE、复数NRMSE、达标信息。
- per_image_metrics.csv：同一图片内跨种子均值和标准差。
- paired_differences.csv：“127.5°减黄金角”的匹配差值；PSNR/SSIM正数、NRMSE负数有利于127.5°。
- mean_paired_differences.csv：先平均图片内种子，再等权平均图片。不把随机种子视为独立物体。
- comparison_images/*.png：真实振幅/相位、两角度重建振幅和复数误差并排图，标题列出图片和条件。固定展示最小seed，不挑最好结果。
- metric_curves/*.png：逐图指标随N和sigma变化的曲线。
- geometry.csv：覆盖率、照明CV、空洞大小、唯一位置数等。
- runs/<id>/history.csv和arrays.npz：迭代指标和复数物体/探针数组。

显示振幅和误差固定色阶[0,1]，超界值在显示时截断，评分不截断。对齐只消除全局复数标量，不校正平移或线性相位坡度。

PSNR/SSIM基于对齐后的振幅，data_range=1；SSIM为7×7均匀窗、样本协方差、K1=.01/K2=.03、丢弃3像素边界。NRMSE评估复数场。phase/mixed根据不同输入振幅a生成phi=1.5*(a-.1)/.9；相位物体取exp(i*phi)，纯相位的振幅指标没有足够区分力，应主要看复数NRMSE。

不按重建效果选图。选图只用输入结构特征，避免先看到哪个角度好再挑样本。候选与入选均属同一合成生成体系，尚不能当作真实样本的独立验证；下一步应加入真实照片、显微样品或独立生成器。

## 续跑

重新执行完全相同命令，跳过已完成任务；未完成任务重跑。不支持迭代中间状态恢复。更换代码、图片、选图数量或参数需要新output目录；程序校验输入文件与源码哈希。

只重绘：原命令加--report-only，自动选图参数须保持一致。

## 验证范围

test_dataset检查分量/孔洞计算、可重复选择、20类覆盖、24张不同文件；test_metrics检查指标的解析性质。experiment.py selftest检查重建数值管线。quick验证完整图表和汇总流程，不提供普适结论或显著性证明。
