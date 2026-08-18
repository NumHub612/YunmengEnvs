"""负责根据配置组装和初始化Python类实例"""

from typing import Any, Dict, Type
import importlib


class ClassBuilder:
    """根据配置动态构建类实例"""

    def build(self, config: Dict[str, Any]) -> Any:
        """解析配置并实例化对应类"""
        class_path = config["class"]
        params = config.get("params", {})

        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**params)


class PipelineAssembler:
    """组装多个组件成执行流水线"""

    def __init__(self, builder: ClassBuilder):
        self.builder = builder

    def assemble(self, pipeline_config: list) -> list:
        """按顺序组装组件"""
        return [self.build(step) for step in pipeline_config]
