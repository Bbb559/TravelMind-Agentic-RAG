"""旅游问题的确定性意图与字段映射。"""

from __future__ import annotations

INTENT_FIELDS: dict[str, tuple[str, ...]] = {
    "itinerary": ("必打卡景点", "交通安排", "实用小贴士"),
    "transport": ("交通安排", "实用小贴士"),
    "attractions": ("必打卡景点", "实用小贴士"),
    "accommodation": ("住宿推荐", "实用小贴士"),
    "food": ("美食推荐", "实用小贴士"),
    "general": ("交通安排", "必打卡景点", "住宿推荐", "美食推荐", "实用小贴士"),
}


def classify_travel_intent(query: str) -> str:
    if any(term in query for term in ("住宿", "酒店", "民宿", "住哪里", "住哪")):
        return "accommodation"
    if any(term in query for term in ("美食", "吃什么", "小吃", "餐厅", "好吃")):
        return "food"
    if any(term in query for term in ("怎么去", "交通", "高铁", "机场", "公交", "打车")):
        return "transport"
    if any(term in query for term in ("必打卡", "景点", "看什么", "值得去")):
        return "attractions"
    if any(
        term in query
        for term in (
            "怎么玩",
            "玩法",
            "安排",
            "路线",
            "几日游",
            "一日游",
            "半天游",
            "半日游",
            "行程",
        )
    ):
        return "itinerary"
    return "general"


def fields_for_intent(intent: str) -> tuple[str, ...]:
    return INTENT_FIELDS.get(intent, INTENT_FIELDS["general"])
