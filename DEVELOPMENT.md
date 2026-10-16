# 开发与测试

## 要求

- Windows 10/11
- Python 3.10+
- Git（仅开发与分发源码时需要）
- 基础依赖：`requirements.txt`（Pillow）

可选依赖清单：

- `requirements-formats.txt`：HEIC/HEIF 与 RAW 解码。
- `requirements-semantic.txt`：Transformers、Safetensors 和 NumPy。PyTorch 需要按本机 CPU/CUDA 环境独立安装。

## 启动

```powershell
python -m pip install -r requirements.txt
python server.py
```

或双击 `run.bat`。默认服务地址为 `http://127.0.0.1:8765`。通过 `LOCAL_IMAGE_LIBRARY_PORT` 可指定端口，通过 `LOCAL_IMAGE_LIBRARY_DATA` 可指定数据目录。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试模块 `tests/test_app.py` 会启动临时 HTTP 服务，创建合成图片和临时数据库，并覆盖扫描、搜索、标签、OCR 文本、操作确认与撤销、设置、备份和 UI 静态资源等流程。

`test_siglip_local_index_and_content_search` 需要完整本地 SigLIP2 模型、PyTorch 和语义依赖；模型加载与索引可能显著拉长测试时间。没有这些依赖时，测试会跳过语义运行部分。

JavaScript 语法快速检查：

```powershell
node --check app.js
```

## 项目文件

```text
.
├── app.js
├── index.html
├── style.css
├── server.py
├── semantic.py
├── requirements*.txt
├── run.bat
├── tests/
└── docs/
```

`.local-models/`、应用数据、Python 缓存和测试产物不应提交到源码仓库。模型权重较大且由使用者从模型源下载。

发布仓库或源码包时只包含源码、测试和项目文档。不得发布模型权重、个人图片、索引数据库、缩略图缓存、令牌、密码、密钥或其他个人隐私文件。仓库 `.gitignore` 对常见模型权重、图片、数据库和凭据文件类型做了排除；每次打包或推送前，仍须检查 `git status`、`git ls-files` 和压缩包清单，确认没有本机数据。

## 源码包与公开发布

发布前确认：

1. 包内没有用户图片、数据库、模型权重、令牌或缓存。
2. README 的支持平台、功能说明和依赖与当前实现一致。
3. 选择适当的许可证。项目没有许可证时，公开代码默认并不授予他人复制、修改或再发布的授权。
4. 在 GitHub 建立目标仓库并确认公开/私有范围，再推送源码。
