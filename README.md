<p align="center">
  <a href="https://github.com/NumHub612/YunmengEnvs">
    <img 
      src="./docs/assets/logo.jpg" 
      onerror="if(!this.dataset.retried){this.dataset.retried='1';this.src='./assets/logo.jpg';}" 
      alt="YunmengEnvs" 
      width="540" 
      height="460" 
    />
  </a>
</p>

---

## 什么是 YunmengEnvs？

[![](https://img.shields.io/badge/license-MIT-red?logo=mit)](./LICENSE) [![](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/downloads/) [![codecov](https://codecov.io/gh/NumHub612/YunmengEnvs/branch/main/graph/badge.svg?token=R5niOoGKl0)](https://codecov.io/gh/NumHub612/YunmengEnvs) [![](https://img.shields.io/badge/Docs-YunmengHome-blue?logo=markdown)](https://NumHub612.github.io/YunmengEnvs/)

[简体中文](README.md) | [English](README_en.md)

`YunmengEnvs` 是一个开源的环境流体力学数值模拟框架。我们不再将 AI 仅作为传统求解器的外挂工具，而是将其与物理机理同等视之，为理解环境流体力学概念、分析现实数据和解决工程问题提供一个高度集成的计算平台。

我们不仅希望提供一个工具，更希望将 `YunmengEnvs` 打造为一个**开放的实验室**，帮助研究者和开发者更方便地尝试新的想法、验证前沿技术、探索机理与数据双驱动的解决方案。

**核心设计理念：AI 原生**

在 `YunmengEnvs` 中，机理项与神经项被视作**同为一等公民的算子**。用户可通过组装比例自由选择“多机理少 AI”或“多 AI 少机理”的求解路径。框架基于以下三条核心设计理念构建：

- 🌊 **可微性是分水岭**
  AI 原生意味着梯度能自然穿透整个求解过程，而非仅在求解器内嵌套或外挂神经网络，整条求解链端到端可训。
- ⚙️ **推理与训练解耦**
  求解过程保持纯粹的前向推理逻辑，不掺杂训练细节；反向传播、损失优化等训练关注点交由独立的机制承担。
- 🔌 **后端无关的抽象**
  框架核心抽象不绑定具体的深度学习框架或数学库，底层可灵活适配 Numpy、PyTorch、JAX 等计算后端环境。

**What's News!**

- 🔥🔥🔥 **[2026-07-22]**: 项目处于验证优化阶段，欢迎大家参与测试！

---

## 如何使用？

---

## 加入我们！

`YunmengEnvs` 是一个开放的实验场，我们欢迎来自流体力学、人工智能、计算数学等各个领域的贡献。无论是提交 Bug、完善文档、实现新的物理格式算子，还是引入前沿的神经算子，都是对项目巨大的支持。

我们希望能够做出一些有趣的尝试和成果，能够将这些想法和经验分享给整个社区。如果你对这个项目感兴趣，欢迎加入我们！

- [开发者指南](Developer.md#developer-guide)
- [参与反馈](https://github.com/NumHub612/YunmengEnvs/issues)
- [参与讨论](https://github.com/orgs/NumHub612/discussions)

感谢所有的贡献者:

<a href="https://github.com/NumHub612/YunmengEnvs/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NumHub612/YunmengEnvs" />
</a>

---
