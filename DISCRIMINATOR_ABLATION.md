# 判别器分支消融实验

## 实验定义

当前 `MultiFreqDiscriminator` 有四个频域判别器分支：

- `mag`：幅度谱（magnitude）
- `phase`：相位谱（phase）
- `radial`：径向频谱（radial spectrum）
- `band`：分频带能量（band energy）

消融采用 4 个分支的全子集组合，理论上一共 16 组。每个变体只跑 Stage 1，并用该变体自己的 Stage-1 最优 checkpoint 做结果对比。不要把原始四分支 checkpoint 直接拿来混用，否则不属于同一训练协议。

## 准备

```bash
cd /media/buu/f9ac1451-f2c8-4f9b-ac35-02ca205e11201/deft_work_folder/DeepfakeBench/aaai27
conda activate deepfake
./verify_snapshot.sh
```

如果 `aaai27` 被移动过，先重新配置数据：

```bash
./configure_data.sh /absolute/path/to/datasets/lmdb
```

默认使用 2 张 GPU，可通过 `NPROC_PER_NODE=1` 或其他数值覆盖；如端口冲突，设置 `MASTER_PORT`。

## 逐个执行

每条命令都会自动执行该变体的 Stage 1，并把结果写入 `logs/training/`：

```bash
./run_discriminator_ablation.sh none
./run_discriminator_ablation.sh mag
./run_discriminator_ablation.sh phase
./run_discriminator_ablation.sh radial
./run_discriminator_ablation.sh band
./run_discriminator_ablation.sh mag_phase
./run_discriminator_ablation.sh mag_radial
./run_discriminator_ablation.sh mag_band
./run_discriminator_ablation.sh phase_radial
./run_discriminator_ablation.sh phase_band
./run_discriminator_ablation.sh radial_band
./run_discriminator_ablation.sh wo_mag
./run_discriminator_ablation.sh wo_phase
./run_discriminator_ablation.sh wo_radial
./run_discriminator_ablation.sh wo_band
./run_discriminator_ablation.sh all
```

对应关系：

| 实验 | 启用的分支 | 需要执行 |
|---|---|---|
| `none` | `none` | `./run_discriminator_ablation.sh none` |
| `mag` | `mag` | `./run_discriminator_ablation.sh mag` |
| `phase` | `phase` | `./run_discriminator_ablation.sh phase` |
| `radial` | `radial` | `./run_discriminator_ablation.sh radial` |
| `band` | `band` | `./run_discriminator_ablation.sh band` |
| `mag_phase` | `mag + phase` | `./run_discriminator_ablation.sh mag_phase` |
| `mag_radial` | `mag + radial` | `./run_discriminator_ablation.sh mag_radial` |
| `mag_band` | `mag + band` | `./run_discriminator_ablation.sh mag_band` |
| `phase_radial` | `phase + radial` | `./run_discriminator_ablation.sh phase_radial` |
| `phase_band` | `phase + band` | `./run_discriminator_ablation.sh phase_band` |
| `radial_band` | `radial + band` | `./run_discriminator_ablation.sh radial_band` |
| `wo_mag` | `phase + radial + band` | `./run_discriminator_ablation.sh wo_mag` |
| `wo_phase` | `mag + radial + band` | `./run_discriminator_ablation.sh wo_phase` |
| `wo_radial` | `mag + phase + band` | `./run_discriminator_ablation.sh wo_radial` |
| `wo_band` | `mag + phase + radial` | `./run_discriminator_ablation.sh wo_band` |
| `all` | `mag + phase + radial + band` | `./run_discriminator_ablation.sh all` |

全部顺序执行：

```bash
./run_all_discriminator_ablations.sh
```

结果目录中的 `val/avg/ckpt_best.pth` 是该变体 Stage 1 的最佳模型。建议记录 Celeb-DF-v1、Celeb-DF-v2 的 AUC、ACC、EER 和 AP，并保持 batch size、随机种子、epoch、数据划分和学习率不变。
