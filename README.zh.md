<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/bentoml/BentoML/assets/489344/d3e6c95d-d224-49a5-9cff-0789f094e127">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/bentoml/BentoML/assets/489344/de4da660-6aeb-4e5a-bf76-b7177435444d">
    <img alt="BentoML: Unified Model Serving Framework" src="https://github.com/bentoml/BentoML/assets/489344/de4da660-6aeb-4e5a-bf76-b7177435444d" width="370" style="max-width: 100%;">
</picture>

## 统一的模型推理服务框架

🍱 使用任意开源或自定义 AI 模型构建模型推理 API 与多模型服务系统。👉 [加入我们的社区论坛](https://forum.modular.com/c/bento/31)！

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202-green.svg)](https://github.com/bentoml/BentoML?tab=Apache-2.0-1-ov-file)
[![Releases](https://img.shields.io/github/v/release/bentoml/bentoml.svg)](https://github.com/bentoml/bentoml/releases)
[![CI](https://github.com/bentoml/bentoml/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bentoml/BentoML/actions/workflows/ci.yml?query=branch%3Amain)
[![Twitter](https://badgen.net/badge/icon/@bentomlai/1DA1F2?icon=twitter&label=Follow)](https://twitter.com/bentomlai)

[ [English](https://github.com/bentoml/BentoML/blob/main/README.md) | 简体中文 ]

## 什么是 BentoML？

BentoML 是一个专为 AI 应用与模型推理优化的在线服务系统构建 Python 库。

- **🍱 轻松为任意 AI/ML 模型构建 API。** 只需寥寥数行代码与标准 Python 类型注解（type hints），即可将任意模型推理脚本转化为 REST API 服务。
- **🐳 极简的 Docker 容器化体验。** 告别依赖地狱！通过简洁的配置即可管理运行环境、依赖项与模型版本。BentoML 可自动生成 Docker 镜像，确保环境可复现，并大幅简化跨环境部署流程。
- **🧭 最大化 CPU/GPU 资源利用率。** 借助动态批处理（dynamic batching）、模型并行、多阶段流水线以及多模型推理图编排等内置服务优化特性，构建高性能推理 API。
- **👩‍💻 高度可定制。** 轻松实现包含自定义业务逻辑、模型推理及多模型组合的专属 API 或任务队列，全面支持各类机器学习框架、模态与推理运行时。
- **🚀 生产级就绪。** 支持在本地高效开发、运行与调试，并可通过 Docker 容器或 Kubernetes 无缝部署至生产环境。

## 快速开始

安装 BentoML：

```
# 需要 Python≥3.9
pip install -U bentoml
```

在 `service.py` 文件中定义 API：

```python
import bentoml

@bentoml.service(
    image=bentoml.images.Image(python_version="3.11").python_packages("torch", "transformers"),
)
class Summarization:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline('summarization', device=device)

    @bentoml.api(batchable=True)
    def summarize(self, texts: list[str]) -> list[str]:
        results = self.pipeline(texts)
        return [item['summary_text'] for item in results]
```

### 💻 本地运行

将 PyTorch 和 Transformers 软件包安装到你的 Python 虚拟环境中：

```bash
pip install torch transformers  # 本地运行所需的额外依赖
```

在本地启动服务代码（默认监听 http://localhost:3000）：

```bash
bentoml serve
```

你将看到如下输出：

```
[INFO] [cli] Starting production HTTP BentoServer from "service:Summarization" listening on http://localhost:3000 (Press CTRL+C to quit)
[INFO] [entry_service:Summarization:1] Service Summarization initialized
```

现在你可以通过浏览器访问 http://localhost:3000 或使用 Python 脚本执行推理：

```python
import bentoml

with bentoml.SyncHTTPClient('http://localhost:3000') as client:
    summarized_text: str = client.summarize([bentoml.__doc__])[0]
    print(f"Result: {summarized_text}")
```

### 🐳 使用 Docker 部署

运行 `bentoml build`，将所需代码、模型与依赖配置打包为一个 Bento（BentoML 中的标准化可部署制品）：

```bash
bentoml build
```

确保 [Docker](https://docs.docker.com/) 正在运行，生成用于部署的 Docker 容器镜像：

```bash
bentoml containerize summarization:latest
```

运行生成的镜像：

```bash
docker run --rm -p 3000:3000 summarization:latest
```

## 示例项目

- 大语言模型（LLMs）：[Llama 3.2](https://github.com/bentoml/BentoVLLM/tree/main/llama3.2-11b-vision-instruct)、[Mistral](https://github.com/bentoml/BentoVLLM/tree/main/ministral-8b-instruct-2410)、[DeepSeek Distil](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-r1-distill-llama3.1-8b-tool-calling) 等。
- 图像生成：[Stable Diffusion 3 Medium](https://github.com/bentoml/BentoDiffusion/tree/main/sd3-medium)、[Stable Video Diffusion](https://github.com/bentoml/BentoDiffusion/tree/main/svd)、[Stable Diffusion XL Turbo](https://github.com/bentoml/BentoDiffusion/tree/main/sdxl-turbo)、[ControlNet](https://github.com/bentoml/BentoDiffusion/tree/main/controlnet) 以及 [LCM LoRAs](https://github.com/bentoml/BentoDiffusion/tree/main/lcm)。
- 向量嵌入（Embeddings）：[SentenceTransformers](https://github.com/bentoml/BentoSentenceTransformers) 与 [ColPali](https://github.com/bentoml/BentoColPali)
- 音频处理：[ChatTTS](https://github.com/bentoml/BentoChatTTS)、[XTTS](https://github.com/bentoml/BentoXTTS)、[WhisperX](https://github.com/bentoml/BentoWhisperX)、[Bark](https://github.com/bentoml/BentoBark)
- 计算机视觉：[YOLO](https://github.com/bentoml/BentoYolo) 与 [ResNet](https://github.com/bentoml/BentoResnet)
- 进阶示例：[函数调用（Function calling）](https://github.com/bentoml/BentoFunctionCalling)、[LangGraph](https://github.com/bentoml/BentoLangGraph)、[CrewAI](https://github.com/bentoml/BentoCrewAI)

查看[完整列表](https://github.com/orgs/bentoml/repositories)获取更多示例代码与使用方式。

## 进阶主题

- [模型组合（Model composition）](https://docs.bentoml.com/en/latest/get-started/model-composition.html)
- [工作进程与模型并行化（Workers and model parallelization）](https://docs.bentoml.com/en/latest/build-with-bentoml/parallelize-requests.html)
- [自适应批处理（Adaptive batching）](https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html)
- [GPU 推理（GPU inference）](https://docs.bentoml.com/en/latest/build-with-bentoml/gpu-inference.html)
- [分布式服务系统（Distributed serving systems）](https://docs.bentoml.com/en/latest/build-with-bentoml/distributed-services.html)
- [模型加载与模型仓库（Model loading and Model Store）](https://docs.bentoml.com/en/latest/build-with-bentoml/model-loading-and-management.html)
- [可观测性（Observability）](https://docs.bentoml.com/en/latest/build-with-bentoml/observability/index.html)

请参阅[官方文档](https://docs.bentoml.com)获取更多教程与指南。

## 社区

欢迎参与并加入我们的[社区论坛 💬](https://forum.modular.com/c/bento/31)，数以千计的 AI/ML 工程师在这里互帮互助、为项目添砖加瓦，并共同探讨 AI 产品的构建之道。

如需报告 Bug 或提交功能请求，请使用 [GitHub Issues](https://github.com/bentoml/BentoML/issues/new/choose)。

### 参与贡献

你可以通过多种方式为项目做出贡献：

- 报告 Bug 并对与你相关的 [Issues](https://github.com/bentoml/BentoML/issues) 点赞（"Thumbs up"）。
- 排查 [Issues](https://github.com/bentoml/BentoML/issues) 并审阅其他开发者的 [Pull Requests](https://github.com/bentoml/BentoML/pulls)。
- 通过提交 GitHub Pull Request 为项目贡献代码或[文档](https://docs.bentoml.com/en/latest/index.html)。
- 查阅[贡献指南](https://github.com/bentoml/BentoML/blob/main/CONTRIBUTING.md)与[开发指南](https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md)了解更多详情。
- 在我们的[论坛](https://forum.modular.com/c/bento/31)中分享反馈并参与路线图讨论。

感谢所有杰出的贡献者！

<a href="https://github.com/bentoml/BentoML/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bentoml/BentoML" />
</a>

### 使用情况追踪与反馈

BentoML 框架会收集匿名使用数据，以帮助社区持续改进产品。仅会上报 BentoML 的内部 API 调用，不包含任何敏感信息（例如用户代码、模型数据、模型名称或堆栈跟踪）。你可以查看用于使用情况追踪的[源码](https://github.com/bentoml/BentoML/blob/main/src/bentoml/_internal/utils/analytics/usage_stats.py)。你可以通过 `--do-not-track` CLI 选项退订使用情况追踪：

```bash
bentoml [command] --do-not-track
```

或通过设置环境变量关闭：

```bash
export BENTOML_DO_NOT_TRACK=True
```

### 开源许可证

[Apache License 2.0](https://github.com/bentoml/BentoML/blob/main/LICENSE)

---

> _注：本中文文档由社区志愿者翻译与维护，旨在降低中文开发者的阅读门槛。如与英文原版存在差异或更新延迟，请以英文原版 [README.md](https://github.com/bentoml/BentoML/blob/main/README.md) 为唯一权威标准（Single Source of Truth）。_
