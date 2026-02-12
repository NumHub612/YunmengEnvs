# Install

`YunmengEnvs` 基于 **Python3.10.14 + CUDA12.4** 环境开发，建议python/cuda环境版本保持一致。

（*以下操作均位于项目根目录下。*）

1. 创建 Conda 环境：

```
conda create -n yun python=3.12.14 -y
conda activate yun
conda install -c conda-forge pymetis -y
```

2. 安装依赖包：

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 运行案例集：

```
python app.py ./benchmarks/burgers_grid2d/links.yml
```

4. 运行测试集：

```
>>> python -m unittest discover -s ./tests -p "test_*.py"  # 运行所有test_开头的测试脚本

>>> python -m unittest -v tests.test_xxx  # 运行指定测试脚本

>>> python -m unittest -v tests.test_xxx.TestXxx.test_xxx  # 运行指定测试对象的测试方法 
```

-------
