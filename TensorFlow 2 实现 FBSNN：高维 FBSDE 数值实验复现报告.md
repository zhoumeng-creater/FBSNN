## TensorFlow 2 实现 FBSNN：高维 FBSDE 数值实验复现报告

- 姓名：周盟															日期：<2025-05-30>

## **1. 研究背景与目标**

本工作基于 Raissi 等人的论文 *Forward-Backward Stochastic Neural Networks: Deep Learning of High-Dimensional Partial Differential Equations*（arXiv:1804.07010）。论文提出的 FBSNN 框架使用 TensorFlow 1 实现。

研究目标是：

1. 用 **TensorFlow 2 / Keras** 在保持数值精度的前提下重写 FBSNN，代码更简洁、可读、易扩展。

2. 复现论文中第 4 节的关键高维基准实验——

   

   - Black-Scholes-Barenblatt（100 维）；
   - Hamilton–Jacobi–Bellman（100 维）；
   - Allen-Cahn（20 维，可选）。

------



## **2. 实验环境**

- Python 版本：3.9

- TensorFlow 版本：2.10.0（带 GPU 支持，CUDA 12.5，cuDNN v8.9.7）

- 主要硬件：NVIDIA 4050 Laptop 单卡，Intel i7-13700HX

- 复现代码仓库地址：https://github.com/zhoumeng-creater/FBSNN.git

  

  - BlackScholesBarenblatt100D.py：Black-Scholes-Barenblatt复现代码
  - AllenCahn20D.py：AllenCahn复现代码
  - HamiltonJacobiBellman100D.py：HamiltonJacobiBellman复现代码
  - figures/：日志与图片

  

> **可复现性保证**：固定 tf.random.set_seed(123)/tf.random.set_seed(SEED) 和 np.random.seed(123)/np.random.seed(SEED)；所有结果均可通过脚本复现。



------



## **3. 关键代码改写要点**

**数值稳定性增强**：

- 同样在这两个文件中，针对精确解 `u_exact` 中的蒙特卡洛模拟，应用了 log-sum-exp 技巧，有效规避了高维计算中指数函数的溢出风险。
- 在 `HamiltonJacobiBellman100D.py`、`AllenCahn20D.py` 及 `BlackScholesBarenblatt100D.py` 中，修改了误差计算方法，对分母（真实值 `Y_test`）接近零的情况进行了保护，避免了相对误差爆炸的问题。
- 在 `FBSNNs.py` 文件的 `train_step` 方法中，新增了梯度裁剪（`tf.clip_by_global_norm`），以防止梯度爆炸，增强了训练过程的稳定性。

**运算效率提升**：

- 在 `FBSNNs.py` 文件的 `loss_function` 中，优化了 Euler 循环，通过预先计算 `sigma_tf` 并复用其结果，减少了不必要的函数调用和重复计算。
- 在 `FBSNNs.py` 中，初始化损失值时，从 Python 浮点数 `0.0` 改为 `tf.constant(0., tf.float32)`，确保张量操作的一致性。
- 同样在 `FBSNNs.py` 的 `fetch_minibatch` 方法中，生成布朗运动增量时直接指定 `astype(np.float32)`，避免了后续可能的类型转换开销。
- 在 `HamiltonJacobiBellman100D.py`、`AllenCahn20D.py` 和 `BlackScholesBarenblatt100D.py` 中，推荐使用 `@tf.function`（及 `jit_compile=True`）装饰器来加速 `train_step` 等关键计算函数，通过图优化和XLA编译提升执行效率。
- 对上述三个高维问题文件，启用混合精度训练（`mixed_float16`），以在兼容硬件上实现显存减半和训练加速。

**训练控制与可复现性改进**：

- 在 `FBSNNs.py` 中，将学习率 `learning_rate` 修改为 `tf.Variable` (self.lr)，使其成为可动态调整的变量，并直接应用于 `tf.keras.optimizers.Adam` 优化器。
- 在 `HamiltonJacobiBellman100D.py` 和 `AllenCahn20D.py` 中，引入了自动学习率调度机制 (`tf.keras.optimizers.schedules.PiecewiseConstantDecay`)，以避免手动分阶段训练时因重新实例化优化器而丢失动量信息。`BlackScholesBarenblatt100D.py` 中也强调了正确更新优化器学习率的必要性。
- 在 `HamiltonJacobiBellman100D.py`、`AllenCahn20D.py` 及 `BlackScholesBarenblatt100D.py` 的主程序入口处，统一添加了 `np.random.seed` 和 `tf.random.set_seed`，以保证实验结果的可复现性。



其中，关于AI建议修改的部分已经做好了标注。

------



## **4. 实验设计与结果**

### **4.1 Black-Scholes-Barenblatt（100 D）**

- **方程**：见论文式 (15)–(17)。

- **网络配置**：layers=[101, 256, 256, 256, 1]；activation=tanh。

- **训练**：批量 256，学习率 1e-3 → 1e-4 余弦退火，总迭代 100 000 步。

- **所得指标**

  

  - 初始值 Y_0：<72.93±0.05>
  - 平均相对误差：<9.7 × 10^{-3}>

- **可视化**

  - 图 1：100 条路径相对误差均值±2σ。见 figures/B-1.png。
  - 图 2：5 条随机路径 Y_t vs 真解。见 figures/B-2.png。

  



### **4.2 Hamilton–Jacobi–Bellman（100 D）**

- **方程**：式 (18)–(19)。
- **网络配置**：layers=[101, 256, 256, 256, 1]。
- **训练**：同上但迭代 80 000 步。
- **结果**
  - Y_0：<4.58±0.02>
  - 平均误差：<1.4 × 10^{-2}>
- **图像**
  - Figures/H-1.png、figures/H-2.png。



### **4.3 Allen–Cahn**

- **网络配置**：layers=[21, 128, 128, 128, 1]。
- **主要结果**
  - Y_0：<0.309>；训练 40 000 步 ≈ 15 min。
  - Figures/ac_traj.png 展示 5 条路径。



------

## **5. 结果分析**

- **准确性**：三组实验的 Y_0 与论文对比，误差均低于 1%。
- **代码简洁度**：Python 文件行数有所增加，但是删除了 tf.placeholder、sess.run、variable_scope 等冗余代码，复用率高。

------

## **6. 结论**

我们成功在 TensorFlow 2 平台下重写 FBSNN 并复现了论文的全部高维基准，验证了算法的有效性。

------

## **7. 复现实验步骤**

1. git clone https://github.com/zhoumeng-creater/FBSNN.git
2. conda env create -f environment.yml && conda activate tf2_fbsnn
3. 下载必要库、TensorFlow等
4. 运行代码文件。

报告图自动保存在 figures/。



------

## **8. 参考文献**

- E, W., Han, J., & Jentzen, A. (2017). Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations. *Communications in Mathematics and Statistics*, *5*(4), 349–380. https://doi.org/10.1007/s40304-017-0117-6
- Han, J., Jentzen, A., & E, W. (2018). Solving high-dimensional partial differential equations using deep learning. *Proceedings of the National Academy of Sciences*, *115*(34), 8505–8510. https://doi.org/10.1073/pnas.1718942115