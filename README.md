# 双自由度云台实时人体跟踪系统

本仓库是《综合设计与实践B》课程大作业项目。项目基于 Orange Pi 5 Pro / RK3588 开发板，实现了一个双自由度云台人体跟踪系统：摄像头采集画面后，系统通过 OpenCV 或 YOLO 检测人体目标，计算目标中心与画面中心之间的偏差，并驱动 PCA9685 舵机控制板调整 pan/tilt 两个舵机，使目标尽量保持在画面中心。

项目同时实现了 Web 监控、语音提示、CSV 指标记录、C/C++ 加速、pymp 并行实验和 RKNN/NPU 部署验证，用于展示完整的嵌入式视觉跟踪流程。

## 项目结构

```text
.
├── tracker/        # 主程序入口和摄像头封装
├── common/         # 配置、限幅函数和运行指标记录
├── inference/      # OpenCV、YOLO ONNX、YOLO RKNN 检测与后处理
├── motion/         # 比例控制器、PCA9685 舵机驱动和云台状态
├── web/            # Web 监控页面、MJPEG 视频流和状态接口
├── voice/          # 目标出现时的语音提示逻辑
├── acceleration/   # C/C++ 扩展、pymp 卷积实验和性能 benchmark
├── tool/           # RKNN 转换、NMS benchmark、舵机扫描等辅助工具
├── scripts/        # 常用演示脚本
├── tests/          # CPU 安全的单元测试
├── report/         # 最终课程报告 PDF
├── result/         # 实验结果 CSV 与逐帧日志
├── model/          # 本地模型文件目录，模型权重不提交
├── audio/          # 语音提示音频资源
├── illustration/   # 报告图表、截图和绘图素材
├── environment.yml # Conda 环境配置
└── pyproject.toml  # Python 包配置和命令行入口
```

## 环境配置

推荐使用 Python 3.10。课程开发环境使用 conda 环境 `b`：

```bash
conda env update -n b -f environment.yml
conda activate b
python -m pip install -e . --no-deps
```

RKNN runtime 与 NPU 驱动依赖开发板环境和 Rockchip 官方 wheel，不建议写死在通用环境文件里。需要运行 RKNN 检测时，应在 RK3588 开发板上单独安装 `rknn-toolkit-lite2` 相关运行库。

## 常用命令

无硬件快速自检：

```bash
python -m tracker.app --mock-servo --no-display --max-frames 5
```

运行单元测试：

```bash
python -m pytest tests -q
```

OpenCV 保底检测与云台跟踪：

```bash
python -m tracker.app --detector face
```

ONNX CPU 检测：

```bash
python -m tracker.app --detector yolo_onnx --model model/yolo11n.onnx --target-class person --mock-servo --no-display
```

RKNN NPU 检测：

```bash
python -m tracker.app --detector yolo_rknn --model model/yolo11n.rknn --target-class person --mock-servo --no-display
```

启动带 Web 监控的跟踪：

```bash
python -m tracker.app --detector face --web-host 0.0.0.0 --web-port 8080 --no-display
```

浏览器访问：

```text
http://<开发板IP>:8080/
```

构建 C 扩展：

```bash
python -m build acceleration
```

运行性能测试：

```bash
python acceleration/benchmark.py --height 160 --width 240 --iterations 3
python tool/benchmark_nms.py --boxes 1200 --iterations 50
```

## 演示脚本

`scripts/` 中保留了一组一行式脚本，便于现场运行：

```bash
bash scripts/run.sh
bash scripts/run_camera.sh
bash scripts/run_rknn.sh
bash scripts/run_tracking.sh
bash scripts/run_web.sh
bash scripts/run_voice.sh
bash scripts/run_servo.sh
bash scripts/run_performance.sh
```

真实云台运行前，应先确认 PCA9685 接线、舵机通道、机械限位和供电状态。若方向相反，可调整 pan/tilt 方向符号；若抖动明显，可减小单帧最大步长或增大死区。

## 实验结果

`result/` 目录只保留实验结果 CSV：

```text
result/
├── required_results.csv
├── tracking_performance.csv
├── bonus_performance.csv
└── rknn_accuracy.csv
```

当前保留结果包括真实云台运行、RKNN 检测、ONNX/RKNN 性能对比、C 扩展加速和 pymp 并行实验等数据。`rknn_accuracy.csv` 记录 1296 帧 RKNN 云台跟踪日志，其中 1096 帧为有效新鲜目标帧；稳定片段第 855--954 帧共 100 帧，平均像素误差为 81.78 px。报告中的图表和文字分析基于这些数据整理。

## 报告

课程报告成版文件为：

```text
report/report.pdf
```

报告图表素材保存在 `illustration/`，主要结果 CSV 保存在 `result/`。LaTeX 编译中间文件和草稿源文件不作为最终提交内容。

## 模型与大文件

模型权重、RKNN 模型、生成视频、大型音频、虚拟环境和机器相关路径不应提交到仓库。需要运行 YOLO ONNX 或 RKNN 检测时，将对应模型放入 `model/` 目录即可。本地常用文件名为 `model/yolo11n.onnx` 和 `model/yolo11n.rknn`，它们已由 `.gitignore` 排除。

## 注意事项

- 真实舵机测试只应在开发板、PCA9685、舵机和外部供电连接正确后进行。
- 日常验证优先使用 `--mock-servo` 和 `--no-display`。
- Web 监控会占用端口，若 8080 被占用可改用其他端口。
- RKNN 检测不会静默回退到 CPU；缺少 runtime 或模型时会直接报错。
- 音频提示默认更适合播放预生成 WAV，避免在 4GB 开发板上实时加载大型 TTS 模型。
