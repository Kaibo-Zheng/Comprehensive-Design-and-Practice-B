# 双自由度云台实时人体跟踪系统 — 最终报告

> 运行环境：Orange Pi 5 Pro（RK3588/RK3588S，约 4GB 内存），conda 环境 `b`（Python 3.10，OpenCV、onnxruntime、rknnlite、pymp、smbus2）。
> 代码目录：仓库根目录的 `app.py`、`camera.py`、`core/`、`inference/`、`motion/`、`web/`、`voice/`（主程序）、`acceleration/`（C 卷积加分模块）、`scripts/`（运行脚本）、`reports/`（兼容日志与报告链接）。
> 本报告引用的数据均来自 `reports/logs/` 下的真实运行日志，详见每节标注。

## 1. 项目背景与任务要求

本项目实现一个基于开发板的双自由度（pan/tilt）摄像头云台人体跟踪系统：USB 摄像头采集图像，检测目标人体，计算图像中心误差，闭环驱动 PCA9685 控制两个舵机，使目标保持在画面中心，同时记录性能指标并提供 Web 远程监控。

大作业要求与本项目实现的对应关系：

| 要求 | 实现方式 | 证据 |
| --- | --- | --- |
| 必须基于开发板运行 | 采集/检测/控制/Web/日志均在 RK3588 板上运行 | 现场运行命令、`reports/logs/*.csv` |
| 必须使用两自由度云台 | PCA9685 驱动 pan(ch0)/tilt(ch1) 两个舵机 | `motion/gimbal.py`、校准记录 |
| 跟踪精度与速度计入评分 | 记录 FPS、各阶段耗时、像素误差、丢失帧、舵机角度 | `metrics.py`、CSV 日志、本报告第 9 节 |
| 加分项 1：C/C++ 编译 whl | `acceleration/` 生成 whl，对比 5x5 卷积 | 第 7 节、`reports/logs/cpp_conv_benchmark.txt` |
| 加分项 2：网络/云端服务 | 主程序内置 Web 监控 + 独立 MJPEG 流 | 第 6 节、`reports/screenshots/web_snapshot.jpg` |
| 加分项 3：任务加速（建议 pymp） | pymp 4 线程加速 2.71x；C 扩展加速 412x | 第 7 节、`reports/logs/cpp_conv_benchmark.txt` |
| 加分项 4：DNN 用 RKNN 部署 | YOLO ONNX→RKNN，RK3588 NPU 推理已跑通 | 第 8、9 节、`reports/logs/p7_yolo_rknn.csv` |

## 2. 系统总体设计

每帧的闭环数据流：

```text
USB 摄像头 → OpenCV 采集 → PersonDetector/Yolo*Detector.detect() → Target
          → TrackingController.update()（像素误差→角度步进）
          → Gimbal.move_to()（角度→脉宽，限位钳制）
          → TrackingMetrics.update()（FPS/误差/耗时/CSV）
          → 叠加显示 / Web 推送
```

模块职责：

| 文件 | 职责 |
| --- | --- |
| `app.py` | 主入口；解析参数→`build_configs()`→`build_detector()` 工厂；串联各模块 |
| `core/config.py` | dataclass 配置（Camera/Detector/Servo/Controller） |
| `camera.py` | 摄像头打开与 V4L2 设置 |
| `inference/detector.py` | `Target` 数据类 + OpenCV 检测器（face/upperbody/hog） |
| `inference/yolo_onnx_detector.py` | onnxruntime CPU YOLO 检测器 |
| `inference/yolo_rknn_detector.py` | RKNN NPU YOLO 检测器（rknn_toolkit_lite2） |
| `inference/postprocess.py` | letterbox、YOLO 输出解码、NMS、目标选择（ONNX/RKNN 共用） |
| `motion/controller.py` | 像素误差→pan/tilt 角度的比例闭环 |
| `motion/gimbal.py` | PCA9685Lite 驱动、mock 模式、安全校准 |
| `core/metrics.py` | FPS(EMA)、各阶段耗时、误差统计、CSV、退出 JSON 摘要 |
| `web/monitor.py` | stdlib `ThreadingHTTPServer`：页面/流/快照/状态 |
| `acceleration/` | C 扩展 5x5 卷积 + Python 对照 + benchmark |

设计要点：配置单向流动（CLI→dataclass→模块，模块不接触 argparse）；检测器统一返回 `Target(bbox, center, label, confidence, stale)`，因此控制器/云台/指标无需知道检测器是 OpenCV、ONNX 还是 RKNN；所有硬件写入都收敛在 `Gimbal` 内（角度→脉宽、限位钳制）。

## 3. 硬件平台与软件环境

- 开发板：Orange Pi 5 Pro（RK3588/RK3588S），可见 GPU/NPU/RGA/MPP 设备节点，NPU 驱动 `v0.9.6`。
- 摄像头：USB UVC（`/dev/video0`，HD 720P），640x480 采集。
- 云台：PCA9685（I2C bus 1，地址 `0x40`，50Hz），pan=通道0、tilt=通道1。
- 软件环境：conda `b`，Python 3.10；OpenCV、onnxruntime 1.23.2（CPUExecutionProvider）、rknnlite 2.3.2、pymp、numpy、smbus2、Adafruit Blinka；C 扩展 `cpp_conv` 已编译为 aarch64 wheel。
- 内存约束：约 4GB。所有重负载命令（模型加载/推理/转换、Web 常驻、benchmark）执行前后检查 `free -h`，并避免并发；详见 `reports/result.md` 各任务的 Memory check。

## 4. 目标检测算法

统一接口：`detect(frame) -> Target | None`。检测失败时返回上一目标并标记 `stale`，最多保持 `hold_frames` 帧，之后返回 `None`（控制器在 `None` 时冻结角度）。多目标按 `area - 0.25*距中心距离` 排序，偏好又大又居中的框。

- **OpenCV（保底）**：默认 Haar 人脸级联；另有 `upperbody`、`hog`。硬件无关、零额外依赖，作为答辩保底路线，CPU 上即可达到可用帧率。
- **YOLO ONNX（CPU 对照）**：onnxruntime 加载 `.onnx`。预处理 letterbox→RGB→NCHW float[0,1]；`postprocess.decode_yolo_output` 兼容 YOLOv5(`5+C`) 与 YOLOv8/v11(`4+C`) 输出，自动按通道方向转置，反 letterbox 映射回原图后做 NMS，再按 `--target-class` 过滤。实测用 `yolo11n`（COCO 80 类，`(1,84,8400)` 输出），跟踪 `person` 类。
- **YOLO RKNN（NPU 目标路线）**：与 ONNX 共用 `postprocess`；预处理改为 letterbox 后的 NHWC uint8（与转换时 `mean=0/std=255` 约定匹配，归一化在 NPU 内部完成）。runtime 不可用时抛出清晰 `RuntimeError`，**不静默回退 CPU**。

关键参数：`--conf-thres`(0.25)、`--iou-thres`(0.45)、`--input-size`(随模型固定输入覆盖)、`--target-class`(person)。

## 5. 云台控制算法

`controller.py` 为纯数学闭环：取图像中心与目标中心的像素误差 `(x_error, y_error)`；落在 `dead_zone` 内则不动；否则 `误差 × gain × sign` 得到角度步进，按 `max_step` 限幅，再叠加到当前 pan/tilt 并钳制到各轴角度范围。控制器跨帧保存当前角度；目标丢失时冻结。

- 方向：`pan_sign` 控水平、`tilt_sign` 控垂直（默认 tilt_sign=-1，即图像向下→云台向下）。方向相反时翻转对应 sign。
- 安全：默认角度范围 pan 45–135°、tilt 60–120°，中心 90°；`max_step` 默认 2°避免突跳；所有角度→脉宽与限位都只在 `Gimbal` 内完成；`--calibrate` 仅做 ±5° 小幅校准且从不自动运行；`--mock-servo` 跳过全部 I2C，仅维护状态，用于无硬件验证。

## 6. Web 远程监控（加分项 2）

`web/monitor.py` 用 Python 标准库 `ThreadingHTTPServer` 实现，无需 Flask/FastAPI。主循环每帧把叠加后的画面 JPEG 编码并连同状态推送给线程安全的 `WebState`；Web 线程只读取最新帧与状态，不参与采集。端点：`/`（页面）、`/stream.mjpg`（MJPEG 流）、`/snapshot.jpg`、`/status.json`（FPS/target/pan/tilt/lost_frames/detector 等）。主程序退出时 `monitor.stop()` 干净关闭并释放端口。

验证（见 `reports/result.md` TASK-003）：`/status.json` 返回完整 JSON；`/snapshot.jpg` 返回 640x480 JPEG（见 `reports/screenshots/web_snapshot.jpg`）；`/stream.mjpg` 输出 multipart 帧；SIGINT 退出后端口释放。另有独立 `web.stream_webcam` 模块作为纯摄像头流备选。启动：`bash scripts/run_web_demo.sh` 或 `--web-host 0.0.0.0 --web-port 8080`。

## 7. C/C++ whl 加速实验（加分项 1）

`acceleration/py_conv.py` 为纯 Python/NumPy 的 5x5 卷积参考实现（边缘复制填充，默认归一化高斯核）；同文件新增 `conv5x5_pymp`，用 `pymp` 把行循环拆到 4 个 worker；`acceleration/src/cconv.c` 是最终采用的优化 C 扩展，提供 `_cconv.conv5x5` 和面向主链路 YOLO 后处理的 `_cconv.nms_xyxy`。`benchmark.py` 先校验输出差异，再报告 ms/frame 与加速比。已重建 wheel `acceleration/dist/cpp_conv-0.1.0-cp310-cp310-linux_aarch64.whl`，可从非源码目录安装导入。

卷积实测（`240x160`，3 次迭代，`reports/logs/cpp_conv_benchmark_fast.txt`）：

| 实现 | ms/frame | 输出最大差异 | 加速比 |
| --- | --- | --- | --- |
| Python/NumPy 参考 | 1493.016 | — | 1x |
| `pymp` 4 线程 | 447.617 | 0 | **3.34x** |
| C 扩展 `_cconv.conv5x5` | 0.915 | 4.6e-5 | **1632.33x** |

输出差异为 0 到 4.6e-5 量级，说明各实现数值一致。`pymp` 能证明 Python 多进程并行有实际加速；C 扩展仍是本功能块的最高性能实现。

YOLO 后处理 NMS 实测（`1200` 个候选框，50 次迭代，`reports/logs/nms_c_benchmark.txt`）：

| 实现 | ms/run | 输出一致性 | 加速比 |
| --- | --- | --- | --- |
| Python NMS | 52.463 | — | 1x |
| C 扩展 `_cconv.nms_xyxy` | 7.093 | 与 Python 保留索引一致 | **7.40x** |

主程序 `inference.postprocess.nms()` 会优先调用 C NMS；扩展缺失时自动回退到 Python NMS。因此 C 加速不仅覆盖作业建议的 5x5 卷积，也直接服务 RKNN/ONNX 目标检测链路。

## 8. RKNN/NPU 部署实验（加分项 4）

YOLO RKNN 已在 RK3588 NPU 上跑通。模型文件位于 `model/yolo11n.rknn`，检测器通过 `rknnlite.api.RKNNLite` 初始化 runtime，推理时不回退 CPU。ONNX CPU 路线保留为对照和 fallback。

板端状态（`reports/logs/rknn_env.txt`，脚本 `scripts/check_rknn_env.sh`）：

```text
NPU device-tree status : okay        RKNPU driver : v0.9.6
RKNN runtime/toolkit   : 2.3.2       rknnlite.api import OK
Model                  : model/yolo11n.rknn
onnxruntime            : 1.23.2 (CPUExecutionProvider)
```

已完成：`inference/yolo_rknn_detector.py`（与 ONNX 同接口、共用后处理、清晰错误、不静默回退）；`scripts/convert_yolo_rknn.sh` / `tool/convert_yolo_rknn.py`（ONNX→RKNN，缺工具链时给出清晰提示而非 traceback）；`scripts/run_rknn_demo.sh` 默认使用 `model/yolo11n.rknn`；`app.py` 退出时释放 NPU 运行时。

验证命令：

```bash
MODEL=model/yolo11n.rknn LOG=reports/logs/p7_yolo_rknn_recheck.csv MAX_FRAMES=5 \
  bash scripts/run_rknn_demo.sh --mock
```

## 9. 跟踪精度与速度测试

测试为开发板本地、`--mock-servo --no-display` 的短时运行（无人在画面内，故 `lost_frames` 高、误差为 0 属预期；目的为验证流水线与采集/推理耗时）。本机 UVC 采集首帧有暖机开销，短跑的 `capture_ms` 偏高，长跑会摊薄。

| 模式 | 检测器 | 设备 | 平均推理耗时 | 平均 FPS | 备注 / 日志 |
| --- | --- | --- | --- | --- | --- |
| OpenCV | face (Haar) | CPU | ≈ 63 ms | ≈ 9.3（30 帧） | 保底；`p0_mock_smoke.csv` |
| YOLO ONNX | person (yolo11n) | CPU | ≈ 137 ms | ≈ 2.0（5 帧，采集受限） | 对照；`p5_yolo_onnx.csv` |
| YOLO RKNN | person | NPU | ≈ 71.4 ms | ≈ 7.0（20 帧，含暖机） | `p7_yolo_rknn.csv`，20/20 帧检测到 person |
| C 5x5 conv | 卷积 | CPU C 扩展 | 0.915 ms/frame（vs 1493.0 Python） | — | 加分项 1；1632x |
| pymp 5x5 conv | 卷积 | CPU 4 worker | 447.617 ms/frame（vs 1493.0 Python） | — | 加分项 3；3.34x |
| C NMS | YOLO 后处理 | CPU C 扩展 | 7.093 ms/run（vs 52.5 Python） | — | 主链路后处理；7.40x |

观察：本系统帧率主要受 UVC 采集暖机和短跑统计影响。YOLO ONNX 单帧推理约 137 ms；RKNN NPU 20 帧实测平均推理约 71.4 ms，并成功检测到 `person`，说明 DNN 已完成板端 NPU 部署。指标 CSV 字段：`timestamp,frame_index,fps,detector,target_found,target_stale,target_label,target_confidence,x_error,y_error,error_norm,pan,tilt,capture_ms,preprocess_ms,inference_ms,postprocess_ms,total_ms,lost_frames`。

## 10. 问题与改进

- **NPU 路线已接通**：当前 RKNN 平均推理约 71.4 ms，仍可通过 INT8 量化、更小输入尺寸或更长稳定运行继续优化。
- **采集耗时主导帧率**：UVC 读取约占单帧大头。改进：用 RGA/MPP 或更低分辨率/MJPG 优化采集，分离采集线程与推理线程。
- **真实云台已确认**：`p8_real_tracking.csv` 记录 150 帧真实运行；pan/tilt 通道、方向、限位已由现场确认。后续可继续优化 Haar 检测稳定性或切换到 RKNN 人体检测。
- **4GB 内存约束**：`torch+ultralytics` 导入即可造成内存紧张（一次运行中观察到），而 onnxruntime 推理本身较轻。改进：检测器只依赖 onnxruntime；重命令用 cgroup（`systemd-run -p MemoryMax=...`）限制爆炸半径并串行执行。
- **加速广度**：已补 `pymp` 对照（3.34x）、C 卷积（1632x）和主链路 C NMS（7.40x）；后续可继续把采集/推理 pipeline 纳入并行化对比。

## 附：复现命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate b
bash scripts/run_opencv_demo.sh --mock     # 保底 demo
bash scripts/run_web_demo.sh               # Web 监控（浏览器访问开发板:8080）
bash scripts/run_onnx_demo.sh --mock       # YOLO ONNX CPU
bash scripts/run_rknn_demo.sh --mock       # YOLO RKNN NPU
bash scripts/benchmark_all.sh              # Python / pymp / C 卷积 benchmark
# 真实云台（仅在板上、确认硬件后）：
python -m app --calibrate
python -m app --detector face --max-step 1.0 --log reports/logs/p8_real_tracking.csv
```
