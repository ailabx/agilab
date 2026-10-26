import sys
import platform
from pathlib import Path
from typing import Dict, Any

import toml

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)

# 配置项——LLM + 数据源 token
_CONFIG_KEYS = ["MODEL", "BASE_URL", "KEY", "TS_TOKEN"]

_DEFAULTS = {
    "MODEL": "glm-5.2",
    "BASE_URL": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    "KEY": "",
    "TS_TOKEN": "",
}


class Config:
    """配置管理类，配置文件存放于 ~/.agilab/config.toml"""

    def __init__(self):
        self._config_file = Path.home() / ".agilab" / "config.toml"
        for key in _CONFIG_KEYS:
            setattr(self, key, _DEFAULTS[key])
        self._init_config()

    def _init_config(self):
        """确保目录存在，有则加载、无则创建"""
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"创建配置目录失败: {e}")
            sys.exit(1)

        if self._config_file.exists():
            self._load()
        else:
            self._save()
            print(f"已创建配置文件: {self._config_file}")

    def _load(self):
        """从文件加载，缺失字段补默认值"""
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                data = toml.load(f)
            for key in _CONFIG_KEYS:
                setattr(self, key, data.get(key, _DEFAULTS[key]))
        except Exception as e:
            print(f"加载配置失败: {e}，使用默认配置")

    def _save(self):
        """保存到文件"""
        data = {key: getattr(self, key) for key in _CONFIG_KEYS}
        try:
            with open(self._config_file, 'w', encoding='utf-8') as f:
                toml.dump(data, f)
        except Exception as e:
            print(f"保存配置失败: {e}")
            raise

    def update(self, **kwargs):
        """更新配置并保存"""
        for key, value in kwargs.items():
            if key in _CONFIG_KEYS:
                setattr(self, key, value)
            else:
                print(f"警告: 未知配置项 '{key}'")
        self._save()

    def reload(self):
        """重新加载配置"""
        self._load()

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return {key: getattr(self, key) for key in _CONFIG_KEYS}

    def get_path(self) -> Dict[str, str]:
        """获取配置文件路径"""
        return {"系统": platform.system(), "配置文件": str(self._config_file)}


config = Config()


if __name__ == "__main__":
    info = config.get_path()
    print(f"系统: {info['系统']}")
    print(f"配置文件: {info['配置文件']}")
    print(f"\n当前配置:")
    for k, v in config.get_all().items():
        shown = "***" if k.endswith("KEY") or k.endswith("TOKEN") and v else v
        print(f"  {k}: {shown}")
