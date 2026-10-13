"""视图层公共工具。"""

from __future__ import annotations

from typing import Optional

from rest_framework.exceptions import ValidationError

from ml.registry import get_architecture

from ..models import TrainedModel


def resolve_trained_model(architecture: Optional[str] = None,
                          model_id: Optional[int] = None) -> TrainedModel:
    """定位推理要用的模型：显式 id 优先，其次该结构的「激活」模型，最后是最新模型。"""
    if model_id:
        model = TrainedModel.objects.filter(pk=model_id).first()
        if model is None:
            raise ValidationError({"trained_model": f"模型 #{model_id} 不存在"})
        return model

    if not architecture:
        raise ValidationError({"architecture": "请指定网络结构或已训练模型"})
    key = architecture.lower()
    if get_architecture(key) is None:
        raise ValidationError({"architecture": f"不支持的模型：{architecture}"})

    queryset = TrainedModel.objects.filter(architecture=key)
    model = queryset.filter(is_active=True).first() or queryset.order_by("-created_at").first()
    if model is None:
        raise ValidationError({
            "architecture": f"「{get_architecture(key)['name']}」还没有训练好的模型，"
                            f"请先到「模型训练」页面训练后再识别"
        })
    return model


def parse_architecture_list(value) -> list[str]:
    """解析对比接口的模型列表参数：支持 JSON 数组、逗号分隔字符串与重复参数。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = []
        for item in value:
            items.extend(parse_architecture_list(item))
        return items
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        import json

        try:
            return parse_architecture_list(json.loads(text))
        except json.JSONDecodeError as exc:
            raise ValidationError({"architectures": f"JSON 解析失败：{exc}"}) from exc
    return [part.strip() for part in text.split(",") if part.strip()]
