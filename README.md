# ModelGym · 模型健身房

面向模型训练的服务器桌面平台。基于 RackTop 1.26.2，保留服务器监控、SSH、项目同步与历史数据，并增加远程训练配置和 GPU 自主排队。

[English](README_EN.md) · [模型训练使用说明](docs/MODEL_TRAINING.md) · [版本更新](docs/Version_overview.md) · [详细版本记录](docs/VERSION_INFOS.md)

## Windows 使用

本分支交付 ModelGym 1.27.0，安装包保存在本地 `artifacts/windows/`。优先使用 `ModelGym_1.27.0_x64-setup.exe`，MSI 供部署使用。运行需要 WebView2；安装程序会按需处理运行时依赖。

从 RackTop 升级时，先退出原 RackTop（包括托盘进程），再运行安装程序。ModelGym 沿用原应用身份、数据库与凭据服务，保留服务器、配置和历史数据。手工解压的旧版目录不会自动删除；安装新版本后请改用 ModelGym 快捷方式。

本分支关闭了原 RackTop 的自动更新源，后续通过 ModelGym 安装包手动升级。

## 模型训练

- 在侧栏“模型训练”选择服务器，配置远程已有的训练脚本和配置文件。
- 默认使用位置式配置文件，也支持参数式 `--config/--cfg`；按服务器预填训练 Conda 环境，支持自定义启动前缀和附加参数。
- 首行可配置 datalist 生成脚本；任务提交后与 GPU 检测同时运行，生成成功且 GPU 空闲后开训，生成失败则停止任务。
- 从所有服务器的 GPU 状态总览指定固定 GPU，可选择繁忙卡；展示最终命令并检查远端路径和环境。
- 新配置默认用 10 GB 空闲显存阈值显示 GPU 状态，实际开训仍检查进程、利用率及稳定时间；训练结束后弹出结果提醒。
- 全部所选 GPU 同时没有计算进程、利用率不高于 10%、空闲显存达标且稳定 30 秒后自动开训。
- GPU 繁忙时先确认后台排队；队列运行在服务器，客户端退出、休眠或 SSH 断开不影响已接受的任务。
- 查看等待原因、状态和日志，取消排队或确认停止训练；失败不会自动重跑。
- 重叠 GPU 按提交顺序协调，互不重叠的卡可并行；重复提交通过任务标识去重。

远端训练要求 Linux、Python 3.8+、NVIDIA 驱动和 `nvidia-smi`。首版支持 NVIDIA 物理 GPU；不提供 MIG、跨服务器训练或集群级 GPU 硬隔离。完整规则与命令示例见 [模型训练使用说明](docs/MODEL_TRAINING.md)。

## 原有能力

保留多服务器 GPU/CPU/内存与进程监控、空闲算力筛选、资源预约提醒、SSH 终端、项目及数据集同步、受管任务日志、历史趋势和告警。原有服务器和历史数据库仍使用兼容路径，名称中的 `racktop` 无需手动更改。

## 开发

技术栈为 Tauri 2、React、TypeScript、Rust 和 SQLite。需要 Node.js 22、Rust stable，以及对应平台的原生构建工具。Windows 需要 MSVC C++ Build Tools、Windows SDK、WebView2 和系统 OpenSSH。

```powershell
npm.cmd ci
npm.cmd run dev
npm.cmd run tauri -- dev
```

本机隔离环境另见未入库的 `.dev/README.md`；使用其中的 `Start.ps1`，会排除工具链目录的 Vite 扫描，并使用独立开发数据库。

验证与 Windows 构建：

```powershell
npm.cmd test -- --maxWorkers=4
npm.cmd run build
cargo test --locked --manifest-path src-tauri/Cargo.toml
npm.cmd run tauri -- build --target x86_64-pc-windows-msvc --bundles nsis,msi
```

远端调度脚本及其本地测试位于 `src-tauri/assets/` 与 `scripts/`。Windows 上的模拟测试不能替代真实 Linux/GPU 集成验证。

## 开源来源

ModelGym 基于 [Tongzh-SEU/RackTop](https://github.com/Tongzh-SEU/RackTop)。保留上游作者与贡献者署名、历史版本记录和原 [GNU GPL v3 许可证](LICENSE)。旧版截图、历史名称及兼容目录沿用原项目资料。
