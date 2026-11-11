"""agents/llm.py - 为 agents 模块提供统一的 LLM 模型创建

复用项目根目录 llm.py 中的 get_model，但允许通过 agents/config.yaml 覆盖模型。
"""

import importlib.util
import sys
from pathlib import Path


def get_model(model_name: str | None = None):
    """获取模型实例。优先使用项目级 get_model，保持全局配置一致。"""
    # 定位项目根目录下的 llm.py
    project_root = Path(__file__).resolve().parent.parent
    llm_path = project_root / "llm.py"

    # 使用 importlib 按文件路径加载，避免与当前 agents/llm.py 同名冲突
    module_name = "_project_root_llm"
    if module_name in sys.modules:
        mod = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, str(llm_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载项目根目录的 llm.py: {llm_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)

    return mod.get_model(model_name)
