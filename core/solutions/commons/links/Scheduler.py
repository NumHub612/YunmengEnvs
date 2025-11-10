# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide linking network analysis and management functionalities.
"""
from core.solutions.standards import ILinkableComponent, IOutput, IInput
from networkx import DiGraph

from typing import Callable
from enum import Enum, auto

from pathlib import Path
import networkx as nx
import yaml

from core.solutions.standards import (
    ILinkableComponent,
    IOutput,
    IInput,
    LinkableComponentStatus,
    LinkableComponentStatusChangeEventArgs,
)
from core.solutions.commons.events import EventManager


class SchedulerStatus(Enum):
    CREATED = auto()
    LOADING = auto()
    READY = auto()  # initialize 完成
    RUNNING = auto()
    PAUSED = auto()  # 遇到断点
    DONE = auto()
    FAILED = auto()
    FINISHING = auto()


class DataFlowMode(Enum):
    PULL = auto()
    PUSH = auto()
    HYBRID = auto()


# ---------- 异常 ----------
class SchedulerError(RuntimeError):
    """调度器级别异常"""

    pass


# ---------- 时间策略 ----------
class ITimeStrategy:
    """时间推进策略接口"""

    def compute_next_timestep(self, comps: list[ILinkableComponent]) -> float:
        """返回下一个全局步长（秒）"""
        raise NotImplementedError

    def is_finished(self, comps: list[ILinkableComponent]) -> bool:
        """全部组件是否已结束"""
        raise NotImplementedError


class Scheduler:
    """The scheduler is responsible for managing the initializing、coupling
    and scheduling of the linking components."""

    # region  组件注册 & 拓扑
    def add_component(self, comp: ILinkableComponent) -> None:
        """注册一个已实例化的组件（可多次调用）"""
        ...

    def load_system(self, system_yaml: Path) -> None:
        """读取 system.yaml, 自动实例化组件并建立拓扑"""
        ...

    @property
    def components(self) -> list[ILinkableComponent]:
        """返回当前已注册组件快照（只读）"""
        ...

    # region 2. 生命周期驱动

    def initialize(self) -> None:
        """顺序或拓扑序 initialize;失败时抛 SchedulerError"""
        ...

    def validate(self) -> dict[str, list[str]]:
        """逐个 validate, 返回 {comp_id: [error, ...]} 映射"""
        ...

    def prepare(self) -> None:
        """prepare 阶段：分配缓存、建立适配器链"""
        ...

    def run(self) -> None:
        """主循环：根据 time_strategy 推进所有组件直至全部 DONE / FAILED"""
        ...

    def finish(self) -> None:
        """优雅收尾：flush 文件、释放句柄、持久化状态"""
        ...

    # region 3. 时间调度策略
    @property
    def time_strategy(self) -> ITimeStrategy:
        """时间推进策略对象（见下文）"""
        ...

    # region 4. 数据调度模式
    @property
    def data_flow_mode(self) -> DataFlowMode:
        """PULL / PUSH / HYBRID"""
        ...

    # region 5. 状态 & 异常
    @property
    def status(self) -> SchedulerStatus:
        """调度器自身状态"""
        ...

    def add_status_listener(
        self, listener: Callable[[LinkableComponentStatusChangeEventArgs], None]
    ) -> None:
        """监听任意组件状态变化"""
        ...

    # region 6. 扩展钩子

    def set_breakpoint(self, time: float, comp_id: str = None) -> None:
        """调试：在指定模拟时间（或组件）暂停"""
        ...

    def snapshot(self, tag: str) -> dict:
        """生成当前全局状态快照（用于断点续算）"""
        ...


class FixedStepStrategy(ITimeStrategy):
    """固定步长策略"""

    def __init__(self, step_sec: float):
        self.step_sec = step_sec

    def compute_next_timestep(self, comps: list[ILinkableComponent]) -> float:
        return self.step_sec

    def is_finished(self, comps: list[ILinkableComponent]) -> bool:
        # 简单判定：所有组件 status == DONE
        return all(c.status == LinkableComponentStatus.DONE for c in comps)


# ---------- 主调度器 ----------
class DefaultScheduler:
    """默认调度器实现（单线程，PULL 模式）"""

    def __init__(self, data_flow_mode: DataFlowMode = DataFlowMode.PULL):
        self._comps: dict[str, ILinkableComponent] = {}
        self._topo: nx.DiGraph = nx.DiGraph()
        self._mode = data_flow_mode
        self._time_strategy: ITimeStrategy = FixedStepStrategy(step_sec=900)
        self._status = SchedulerStatus.CREATED
        self._evt_mgr = EventManager()
        self._breakpoints: set[float] = set()
        self._pause_req = False

    # ---------- 1. 外部唯一入口 ----------
    def load_system(self, system_yaml: Path) -> None:
        self._status = SchedulerStatus.LOADING
        cfg = yaml.safe_load(system_yaml.read_text(encoding="utf-8"))
        spec = cfg["spec"]

        # 1) 实例化所有组件
        for comp_item in spec["components"]:
            comp_path = Path(system_yaml.parent / comp_item["componentFile"]).resolve()
            comp = self._instantiate_from_yaml(comp_path)
            self.add_component(comp)

        # 2) 建立拓扑 & 适配器
        for link in spec.get("links", []):
            self._establish_link(link)

        self._status = SchedulerStatus.READY

    # ---------- 2. 工具：yaml -> 组件 ----------
    def _instantiate_from_yaml(self, yaml_path: Path) -> ILinkableComponent:
        """反射创建组件并注入参数/IO"""
        import importlib

        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        meta = cfg["metadata"]
        spec = cfg["spec"]

        # 反射 modelClass
        cls_path = spec.get("modelClass")
        if not cls_path:
            raise SchedulerError(f"modelClass missing in {yaml_path}")
        module_name, cls_name = cls_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        comp: ILinkableComponent = cls()

        # 注入 arguments
        for k, v in spec.get("arguments", {}).items():
            comp.arguments[k].value = v.get("value")

        # 动态创建 inputs/outputs
        comp.setup(
            inputs_config=spec.get("inputs", []), outputs_config=spec.get("outputs", [])
        )
        return comp

    # ---------- 3. 工具：建立 provider-consumer ----------
    def _establish_link(self, link: dict[str, any]) -> None:
        """link = {provider: "cid/out_id", consumer: "cid/in_id", adapters: [...]}"""
        prov_cid, prov_out_id = link["provider"].split("/", 1)
        cons_cid, cons_in_id = link["consumer"].split("/", 1)

        prov_comp = self._comps[prov_cid]
        cons_comp = self._comps[cons_cid]

        # 找到输出/输入对象
        prov_out = next(o for o in prov_comp.outputs if o.id == prov_out_id)
        cons_in = next(i for i in cons_comp.inputs if i.id == cons_in_id)

        # 适配器链（暂留空，后期扩展）
        adapters_cfg = link.get("adapters", [])
        if adapters_cfg:
            # TODO: 实例化适配器并挂到 prov_out.adapters
            pass

        # 自动绑定
        cons_in.provider = prov_out

    # region 1. 组件注册
    def add_component(self, comp: ILinkableComponent) -> None:
        if comp.id in self._comps:
            raise SchedulerError(f"duplicate component id: {comp.id}")
        self._comps[comp.id] = comp
        self._topo.add_node(comp.id, comp=comp)

    def load_system(self, system_yaml: Path) -> None:
        """占位：未来解析 system.yaml 并自动 add_component + 建立拓扑"""
        raise NotImplementedError("yaml loader not implemented yet")

    @property
    def components(self) -> list[ILinkableComponent]:
        return list(self._comps.values())

    # region 2. 生命周期
    def initialize(self) -> None:
        self._status = SchedulerStatus.READY
        for cid in self._topo_order():
            comp = self._comps[cid]
            comp.initialize()
            if comp.status == LinkableComponentStatus.FAILED:
                raise SchedulerError(f"{cid} initialize failed")

    def validate(self) -> dict[str, list[str]]:
        errs = {}
        for cid, comp in self._comps.items():
            errs[cid] = comp.validate()
        return errs

    def prepare(self) -> None:
        for cid in self._topo_order():
            self._comps[cid].prepare()

    def run(self) -> None:
        self._status = SchedulerStatus.RUNNING
        self._pause_req = False
        while not self._time_strategy.is_finished(self.components):
            if self._pause_requested():
                self._status = SchedulerStatus.PAUSED
                return
            dt_sec = self._time_strategy.compute_next_timestep(self.components)
            for cid in self._topo_order():
                comp = self._comps[cid]
                if comp.status == LinkableComponentStatus.DONE:
                    continue
                if self._mode == DataFlowMode.PULL:
                    self._pull_update(comp, dt_sec)
                else:
                    raise SchedulerError("PUSH/HYBRID not implemented yet")
                if comp.status == LinkableComponentStatus.FAILED:
                    self._status = SchedulerStatus.FAILED
                    return
        self._status = SchedulerStatus.DONE

    def finish(self) -> None:
        self._status = SchedulerStatus.FINISHING
        for cid in reversed(self._topo_order()):  # 逆序释放
            self._comps[cid].finish()
        self._status = SchedulerStatus.DONE

    # region 3. 属性
    @property
    def status(self) -> SchedulerStatus:
        return self._status

    @property
    def time_strategy(self) -> ITimeStrategy:
        return self._time_strategy

    @property
    def data_flow_mode(self) -> DataFlowMode:
        return self._mode

    # region 4. 钩子
    def set_breakpoint(self, time: float, comp_id: str = None) -> None:
        self._breakpoints.add(time)

    def add_status_listener(
        self, listener: Callable[[LinkableComponentStatusChangeEventArgs], None]
    ) -> None:
        # 空壳实现
        pass

    def snapshot(self, tag: str) -> dict[str, str]:
        """返回每个组件 keep_current_state() 的 id"""
        return {
            cid: str(comp.keep_current_state()) for cid, comp in self._comps.items()
        }

    # region 5. 内部辅助
    def _topo_order(self) -> list[str]:
        try:
            return list(nx.topological_sort(self._topo))
        except nx.NetworkXError as e:
            raise SchedulerError("cycle detected") from e

    def _pull_update(self, comp: ILinkableComponent, dt: float):
        # PULL 模式：让组件自己更新
        comp.update([])

    def _pause_requested(self) -> bool:
        # 简单示例：只要遇到断点时间即暂停
        # 实际可扩展为外部信号
        return False
