from agno.models.openai.like import OpenAILike

from config import config

# glm-5.2 默认开启 Thinking 模式，每次请求先 think 再出正文，耗时 1-5s 且不稳定
# 用 extra_body 关闭 thinking，首 token 从 ~5s 降到 ~0.5s
_NO_THINK = {"thinking": {"type": "disabled"}}


def get_model(model_name=None):
    """获取 LLM 模型实例，配置从 config.toml 加载"""
    return OpenAILike(
        id=model_name or config.MODEL,
        api_key=config.KEY,
        base_url=config.BASE_URL,
        extra_body=_NO_THINK,
    )
