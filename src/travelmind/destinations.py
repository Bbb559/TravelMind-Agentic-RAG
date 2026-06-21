"""TravelMind 路由与 GraphRAG coverage 共用的目的地实体词典。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DestinationEntity:
    canonical: str
    aliases: tuple[str, ...]
    scope: str
    ancestors: tuple[str, ...] = ()


_PLAIN_DESTINATIONS = (
    "北京",
    "上海",
    "天津",
    "重庆",
    "河北",
    "山西",
    "辽宁",
    "吉林",
    "黑龙江",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "海南",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "内蒙古",
    "广西",
    "西藏",
    "宁夏",
    "新疆",
    "大理",
    "丽江",
    "香格里拉",
    "成都",
    "西安",
    "南京",
    "阳朔",
    "张家界",
    "凤凰",
    "平遥",
    "荔波",
    "黄果树",
    "桂林",
    "杭州",
    "苏州",
)

DESTINATION_ENTITIES = (
    DestinationEntity(
        canonical="荔波小七孔",
        aliases=("荔波小七孔", "小七孔景区", "小七孔"),
        scope="荔波",
        ancestors=("荔波", "贵州"),
    ),
    DestinationEntity(
        canonical="荔波大七孔",
        aliases=("荔波大七孔", "大七孔景区", "大七孔"),
        scope="荔波",
        ancestors=("荔波", "贵州"),
    ),
    DestinationEntity(
        canonical="双廊",
        aliases=("双廊",),
        scope="大理",
        ancestors=("大理", "云南"),
    ),
    DestinationEntity(
        canonical="都江堰",
        aliases=("都江堰景区", "都江堰"),
        scope="成都",
        ancestors=("成都", "四川"),
    ),
    DestinationEntity("荔波", ("荔波",), "荔波", ("贵州",)),
    DestinationEntity("成都", ("成都",), "成都", ("四川",)),
    DestinationEntity("大理", ("大理",), "大理", ("云南",)),
    *(
        DestinationEntity(canonical=name, aliases=(name,), scope=name)
        for name in _PLAIN_DESTINATIONS
        if name not in {"荔波", "成都", "大理"}
    ),
)

_BY_CANONICAL = {entity.canonical: entity for entity in DESTINATION_ENTITIES}


def match_destination_entities(query: str) -> list[str]:
    """按最长别名优先提取规范实体，并去掉已被具体景点覆盖的上级地域。"""
    candidates: list[tuple[int, int, DestinationEntity]] = []
    for entity in DESTINATION_ENTITIES:
        for alias in entity.aliases:
            start = query.find(alias)
            if start >= 0:
                candidates.append((start, -len(alias), entity))
                break

    candidates.sort(key=lambda item: (item[0], item[1], item[2].canonical))
    selected: list[DestinationEntity] = []
    seen: set[str] = set()
    for _, _, entity in candidates:
        if entity.canonical in seen:
            continue
        selected.append(entity)
        seen.add(entity.canonical)

    covered_ancestors = {
        ancestor
        for entity in selected
        for ancestor in entity.ancestors
    }
    return [
        entity.canonical
        for entity in selected
        if entity.canonical not in covered_ancestors
    ]


def destination_scopes(entities: list[str]) -> set[str]:
    return {
        _BY_CANONICAL.get(entity, DestinationEntity(entity, (entity,), entity)).scope
        for entity in entities
    }


def entity_is_mentioned(entity: str, text: str) -> bool:
    definition = _BY_CANONICAL.get(entity)
    aliases = definition.aliases if definition is not None else (entity,)
    return any(alias in text for alias in aliases)


def entity_aliases(entity: str) -> tuple[str, ...]:
    definition = _BY_CANONICAL.get(entity)
    return definition.aliases if definition is not None else (entity,)
