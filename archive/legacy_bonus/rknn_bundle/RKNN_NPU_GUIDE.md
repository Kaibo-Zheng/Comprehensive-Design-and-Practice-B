# RKNN / NPU 部署说明书（RK3588 + YOLO11n）

把 YOLO11n 部署到 RK3588 的 NPU，让人体检测从 CPU 的 **~117 ms/帧** 降到 **~15–40 ms/帧**。
流程三段：**① 在你的 x86 机器上把 ONNX 转成 .rknn → ② 拷回开发板 → ③ 板上装运行时并跑**。

> 转换必须在 x86 上做（`rknn-toolkit2` 只有 x86_64 版）。开发板只负责“运行”转好的 `.rknn`。

---

## 0. 开发板现状（已替你查好）

| 项目 | 值 |
| --- | --- |
| 板上运行时 `/usr/lib/librknnrt.so` | **1.4.0** (a10f100eb@2022-09-09) |
| RKNPU 内核驱动 | v0.9.6 |
| 板上 Python / 架构 | **3.10 / aarch64**（conda 环境 `b`） |
| 模型 | `yolo11n`，输入 `[1,3,640,640]` 固定，输出 `[1,84,8400]` |

---

## 1. 版本黄金法则（最重要，先读）

RKNN 这三个组件的版本**必须一致**（主.次版本相同）：

```
x86:  rknn-toolkit2           (转换 .rknn)
板上: rknn_toolkit_lite2      (Python 运行时)
板上: /usr/lib/librknnrt.so   (底层 C 运行时)   <- 现在是 1.4.0
```

不一致的典型症状（板上运行时会直接报）：

```
RKNN Model version: x.x.x not match with rknn runtime version: y.y.y
```

看到这句 = 版本没对齐，按本节把三者统一。

### 选一条路线

- **路线 A — 改动最小（匹配现有 1.4.0）**：x86 装 `rknn-toolkit2 1.4.0`，板上装 `rknn_toolkit_lite2 1.4.0`，板上 `librknnrt.so` 不动（已是 1.4.0）。
  风险：1.4.0 是 2022 年的老版本，且其 `lite2` 可能没有 cp310 轮子（板上 Python 是 3.10）。若安装不上或转换报算子不支持 → 转路线 B。
- **路线 B — 推荐（升级到 2.x）**：x86 装 `rknn-toolkit2 2.x`，板上装 `rknn_toolkit_lite2 2.x`，并把板上 `librknnrt.so`（+ `rknn_server`）也换成同一 2.x。
  好处：2.x 对 Python 3.10 轮子覆盖好、算子支持全，YOLO11 成功率更高。

> 建议：**先按路线 A 试**（零板上运行时改动）。一旦转换或运行报错，再升级到路线 B。第 8 节排错表会告诉你何时该升。

---

## 2. 一次性下载（在 x86 上执行）

开发板上已经备好 `rknn_bundle/`，里面有：

```
rknn_bundle/
├── yolo11n.onnx            # 已重导出为 opset=12 + 固定 640（可直接喂 RKNN，省去你装 ultralytics）
├── convert_yolo_rknn.py    # 转换脚本
└── RKNN_NPU_GUIDE.md       # 本说明书
```

在你的 x86 机器上一条命令拉走（`<板IP>` 换成开发板地址）：

```bash
scp -r root@<板IP>:/root/ws/rknn_bundle ./rknn_bundle
cd rknn_bundle
```

另外要从 Rockchip 官方仓库拿工具（第 3、6 步），仓库地址：

```
https://github.com/airockchip/rknn-toolkit2
```

目录结构（记住这几个路径）：

```
rknn-toolkit2/packages/            # x86 转换工具 wheel（rknn_toolkit2-*.whl）
rknn-toolkit-lite2/packages/       # 板上运行时 wheel（rknn_toolkit_lite2-*.whl）
rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so   # 板上底层 .so（路线 B 用）
rknpu2/runtime/Linux/rknn_server/aarch64/usr/bin/       # 板上 rknn_server（路线 B 用）
```

---

## 3. x86：安装 rknn-toolkit2

建议用独立 conda/venv，避免污染。x86 Python 版本要和 wheel 的 `cpXY` 对上。

```bash
# 克隆仓库（或网页下载对应版本的 zip）
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2

# 路线 A：用 1.4.0 的 wheel（在该 tag/release 的 packages/ 下）
#   git checkout v1.4.0   然后挑 rknn_toolkit2-1.4.0-cp3X-...x86_64.whl
# 路线 B：用最新 2.x 的 wheel（packages/ 下，挑匹配你 Python 的）
pip install ./rknn-toolkit2/packages/rknn_toolkit2-<版本>-cp3X-cp3X-linux_x86_64.whl
# 该 wheel 的依赖（numpy/onnx 等）可能要 pip install -r packages/requirements_cp3X-*.txt
```

验证：

```bash
python -c "from rknn.api import RKNN; print('toolkit OK')"
```

---

## 4. x86：把 ONNX 转成 RKNN

在 `rknn_bundle/` 目录里运行我们的脚本：

```bash
# (1) 浮点模型 —— 最省事，先用这个跑通
python convert_yolo_rknn.py --onnx yolo11n.onnx --output yolo_person.rknn --target rk3588

# (2) INT8 量化 —— NPU 更快、更省内存（可选，建议跑通(1)后再做）
#   先准备 dataset.txt：每行一张有代表性的图片路径（20~100 张、最好含人）
python convert_yolo_rknn.py --onnx yolo11n.onnx --output yolo_person.rknn --target rk3588 --dataset dataset.txt
```

注意：

- 脚本默认 `mean=0,0,0 / std=255,255,255`，**不要改**。板上检测器 `yolo_rknn_detector.py` 正是按这个约定喂 **letterbox 后的 NHWC uint8 RGB**；改了归一化就要同步改检测器预处理。
- 成功后得到 `yolo_person.rknn`。脚本里若 `rknn-toolkit2` 没装好会给清晰提示（exit 3）而不是堆栈。

---

## 5. 把 .rknn 拷回开发板（在 x86 上执行）

```bash
scp yolo_person.rknn root@<板IP>:/root/ws/models/yolo/yolo_person.rknn
```

---

## 6. 开发板：安装运行时 rknn_toolkit_lite2

从同一个仓库 `rknn-toolkit-lite2/packages/` 里，挑 **cp310 + aarch64** 且**版本与第 3 步 toolkit 相同**的 wheel，拷到板上安装：

```bash
# 板上（conda 环境 b）
/root/miniconda3/envs/b/bin/python3 -m pip install \
  rknn_toolkit_lite2-<版本>-cp310-cp310-linux_aarch64.whl

# 验证能导入（板上检测器就是用它）
/root/miniconda3/envs/b/bin/python3 -c "from rknnlite.api import RKNNLite; print('lite runtime OK')"
```

**仅路线 B 需要**：把底层运行时也升到同版本（否则会报第 1 节的 version not match）：

```bash
# 仓库 rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
sudo cp librknnrt.so /usr/lib/librknnrt.so
# 若仓库附带 rknn_server，也更新并重启（多数 yolo demo 不强依赖，但建议一并更新）
#   sudo cp rknn_server /usr/bin/ ; sudo restart_rknn.sh  (或重启板子)
```

---

## 7. 开发板：跑 NPU 检测

```bash
cd /root/ws

# (a) 先 mock 验证流程（不动舵机）
bash scripts/run_rknn_demo.sh --mock

# (b) 真实云台跟踪
bash scripts/run_rknn_demo.sh

# 等价完整命令：
/root/miniconda3/envs/b/bin/python3 -m gimbal_tracker.app \
  --camera /dev/video0 --detector yolo_rknn \
  --model models/yolo/yolo_person.rknn --target-class person \
  --mock-servo --no-display --max-frames 30 \
  --log reports/logs/p7_yolo_rknn.csv
```

看退出时 JSON 摘要里的 `avg_inference_ms`——应当远低于 CPU 的 ~117 ms。把这个数填进 `reports/result.md` 性能表的 “YOLO RKNN / NPU” 行，加分项 4 就完成了。

---

## 8. 排错表

| 症状 | 原因 | 解决 |
| --- | --- | --- |
| `onnx` 加载报 opset 不支持 | toolkit 太老 | 用 bundle 里已导出的 opset12 `yolo11n.onnx`（已处理）；或升 toolkit |
| `build` 报 unsupported op / 某层失败 | 老 toolkit 对 YOLO11 个别算子支持不全 | **转路线 B（2.x）** |
| 板上 `Model version ... not match runtime version` | `.rknn` 与板上 `librknnrt.so` 版本不一致 | 三组件版本对齐（第 1 节）；路线 B 记得换 `.so` |
| `init_runtime failed` | 驱动/权限/server | 确认 `/dev/dri/renderD128` 存在、用 root、（路线B）`rknn_server` 已更新 |
| 板上 `import rknnlite` 失败 | lite2 没装/版本/Python 不匹配 | 第 6 步装对 **cp310 aarch64 同版本** wheel；1.4.0 若无 cp310 轮子就走路线 B |
| 检测框乱/全无 | 量化精度或预处理不一致 | 先用浮点（不量化）验证；确认没改 mean/std |
| x86 装 toolkit 报 Python 不匹配 | wheel 的 cpXY 与你 Python 不符 | 用对应 Python 建 venv，或换匹配的 wheel |

---

## 9. 内存提示（4GB 板）

NPU 推理本身轻，但**别和 Web 常驻服务、benchmark、ONNX 推理同时跑**。验证统一用 `--mock-servo --no-display --max-frames 30`。装 wheel/转换前后看一眼 `free -h`。

---

## 10. 完成后回填

转换+运行成功后：

1. `reports/result.md` 把 TASK-007 从 blocked 改为 done，填上 NPU 的 `avg_inference_ms` 与 FPS。
2. `reports/final_report.md` 第 9 节性能表把 “YOLO RKNN / NPU / 待测” 换成实测值，第 8 节标注 NPU 路线已打通。
3. 如愿意，跑一组 CPU vs NPU 对比（同样 30 帧），把加速比写进报告——这是加分项 4 最直接的证据。

> 卡住了把**完整报错**贴回来（尤其是 toolkit 的 build 日志或板上的 version-not-match 那行），我帮你判断该锁定哪个版本或转路线 B。
