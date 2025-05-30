# Developer Guide

云梦环境项目旨在用于学术以及教学用途，因此在设计各种算法和方案的实现细节时，编码的清晰度是重中之重。  
尽管如此，它的所有数值和算法在很多方面都与面向工业的 CFD 代码中使用的数值和算法相似，并鼓励详尽的文档说明。

+ [Tip1-关于模型开发过程和文档：一个待完善的开发建议](#关于模型开发过程和文档一些待完善的建议)
+ [Tip2-关于项目的发布流程](#关于项目的发布流程)
+ [Tip3-关于项目的编码风格](#关于项目的编码风格)


---------------------------------------------------------------------------------

## 关于模型开发过程和文档：一些待完善的建议

一个模型的开发过程可能会经历以下几个阶段：

**1st: 第一阶段**

提出模型想法，说明开发目的或意义，提供一些基础开发资料、方向或思路；通常来说，`1st`仍处于前期讨论阶段，可以在项目仓库、知乎、课题小组或者其他平台提出，  
作为一个 `issue`、一个 `提问`、一个 `报告`、一个 `动态` 发布。

**2nd: 第二阶段**

设计模型算法，解释理论依据，提供算法详细设计（计算链路闭环）。

这个阶段可能主要聚焦在如何使想法落地，怎么样来设计一个完整的计算逻辑或计算链路？  
这个阶段可能还需要初步的原型实现，敲代码和推公式交织进行。

**3rd: 第三阶段**

实施开发方案，定义模型数据和接口，完成模块开发和单元测试。

进入 `3rd` 阶段通常表明模型的算法已经基本稳定，开始工程项目开发。一般的建议是，着手编码前梳理功能模块关系，确定模块集成接口，设计数据结构；同时，在开发过程中遵循一套代码规范和格式化方案，并及时完成单元测试、集成测试，注意代码覆盖度。

**4th: 第四阶段**

通过模型评估方案，测试模型准确性和稳定性，测试参数敏感性。
  
在可靠的单元测试基础上，进一步地进行相对充分的模型用例测试(需要覆盖可预见的常见场景和部分极端场景)，测试模块的稳定性、适用性，分析模块的参数敏感性，进一步地测试模块的准确性及性能表现。

**模型文档**

为了提高模型和项目的可靠性和可维护性，一个模型组件的文档建议包含：

+ `算法详细设计`，说明算法原理和前提假设，提供完整的计算过程以及涉及的变量定义；
+ `模型接口说明`，说明组件接口，包括接口参数、接口功能和可能的异常等；
+ `测试评估报告`，提供各类测试结果分析报告；
+ `模型操作手册`，说明模型的输入输出格式，演示常见场景下组件使用。

聚焦于模块的可靠性，`测试评估报告`中应该包括模型与现实的一致性、参数及前提假设的影响、参数真值以及其分布的影响分析，提供模型校核报告、参数敏感性评估报告、网格相关性评估报告，数值算法稳定性评估报告，模型性能评估报告。

[<i class="fa fa-home"></i>](#developer-guide)

---------------------------------------------------------------------------------

## 关于项目的发布流程

项目暂定发布在 GitHub；采用 Git 作为协作开发和版本控制工具；采用 Git Flow 开发模式；版本号采用 `year.month.patch` 格式。  

**功能提交**

* 提交频率

每个提交应该只包含一个逻辑上的更改或修复，这样可以更容易追踪和理解每个提交的意图。
建议将每个commit用时控制在 3 小时内，鼓励提高提交频率。
避免将多个不相关的更改混合在一个提交中，以免给代码审查和版本控制带来困扰。

* 提交格式

提交信息的格式通常是：“[类型]: 描述”。

`类型`指这个提交所属类别，可以是 `feat、fix、docs、style、refactor、test、chore` 等。  
`描述`是对提交的简短描述，应尽量清晰明了，突出关键信息。

* 提交内容

提交信息应该描述清楚修改的内容，不要使用模糊的词汇。
尽量提供一些上下文信息，例如为什么做出这个更改、解决了什么问题、有什么影响等。
如果有关联的问题（如`GitHub Issue`等）或任务，可以在提交信息中引用相关的编号。

**Github原生开发**

* Actions 和 Security

通过 github actions 和 security（code scanning）实现 linux、windows 平台下的 ci 方案。

* Issues

通过 github issues 登记开发工作（保留feat、bug等开发足迹）。

* Discussions

通过 github discussions 进行团队沟通交流。
通过 issues 当然也可以讨论，不过更多是已确定的 feat 或 bug。

* Projects

通过 github projects 进行开发计划和进度管理，控制版本发布 。


[<i class="fa fa-home"></i>](#developer-guide)

---------------------------------------------------------------------------------

## 关于项目的编码风格
  
在开发过程中培养代码工程素养，对代码 “有讲究”，有助于我们开发更可靠和好维护的模型模块：

+ 统一的命名逻辑、良好的封装隔离、清晰的数据流，能帮助控制代码的复杂度。
+ 美观、符合直觉、直白的代码。
+ 好的代码读起来有一种韵律感，能让人一天都很开心。

对于单个概念，定义好其内涵外延；对于多个概念，梳理好其层次关系。映射到代码上，就是不同类的职责清晰划分。

但需要声明，不同的阶段对讲究程度要求是不一样的，如何抉择呢？这里可以引入一个参考度量：**生命周期**。
生命周期越长的代码，一定要写的越干净；临时使用代码，比如原型、脚本，就可以不讲究一些。
反过来，也正是干净的代码才能成就超长的生命周期。

在软件项目开发中，代码的规范性主要包括几个方面：

+ 代码重复率；
+ 命名的规范；
+ 单元测试的覆盖率；
+ 日志打印的规范性。


**编码风格**

绿洲项目中主要参考 [Google Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules.html)，主要追求：

1. 一致性：遵循统一的编码风格，使得代码易于阅读和理解。

2. 易读性：代码应该易于理解，易于维护。


**代码注释**

我们完全不介意你的注释是代码的2倍、3倍 ...，你的代码、你的成果，你想说什么、想说多少都可以。


**日志**

良好的系统，可以通过日志进行问题定为。除了在本地代码上复现、调试外，还要能够通过丰富合理的日志信息还原问题现场，发现错误位置和原因。

*为什么打日志*

+ 跟踪程序的警告和错误，标识程序运行中的危险操作、错误操作；
+ 跟踪崩溃bug；
+ 跟踪性能下降的问题范围，通过日志提供的详细执行时间记录找出应用的性能瓶颈；
+ 跟踪操作流程，获取操作发生的具体环境、操作的结果；

*什么时候该打日志？*

1. 经常以功能为核心进行开发，应该在提交代码前，可以确定通过日志可以看到整个流程；

*日志分级*

1. `ERROR`。该级别日志发生时，已经影响了用户的正常使用，通常程序抛错、中止。主要类型有：
    + 所有第三方对接的异常(包括第三方返回错误码)
    + 所有影响功能使用的异常

2. `WARN`。该级别日志不应该出现但是不影响程序继续运行的问题。主要类型有：
    + 异常：不明确异常，只进行了简单的捕获抛出，需要打印这种笼统处理的异常
    + 有容错机制的时候出现的错误情况
    + 找不到配置文件，但是系统能自动创建配置文件
    + 性能即将接近临界值的时候
    + 非预期执行：为程序在“有可能”执行到的地方打印日志

3. `INFO`。该级别日志主要用于记录系统运行状态、等信息，常用于反馈系统当前状态给用户。主要类型有：
    + 不符合业务逻辑预期：打印关键的参数
    + 系统模块的入口与出口处：可以在重要方法级或模块级记录输入与输出    
    + 对外提供的接口入口处：打印接口的唯一标识和简短描述，并且要将传入的参数原样打印
    + 调用其它系统接口的前后：打印所调用接口的系统名称/接口名称和传入参数/响应参数
    + 服务状态变化：程序中重要的状态信息的变化应该记录下来
    + 一些可能很耗时的业务处理：批处理，IO操作

4. `DEBUG`。该级别日志的主要作用是对系统每一步的运行状态进行精确的记录。(该级别的日志一般用于开发调试或测试环节，不建议在生产环境中开启)，尽量做到：
    + 开发人员和测试人员都能看懂
    + 通过阅读DEBUG级别的日志后不需要重现问题，就能准确的定位解决问题


[<i class="fa fa-home"></i>](#developer-guide)

---------------------------------------------------------------------------------

