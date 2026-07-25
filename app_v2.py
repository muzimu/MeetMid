"""
智能中间点推荐系统 v2 — 多 Agent 架构
======================================
Agent1（规划）:  LLM 理解需求 → 结构化搜索参数
Agent2（搜索）:  LLM + 受控工具 → 搜索候选地点（上下文精简）
路线计算:        纯 Python 直接调高德 API → 全员分别计算
Agent3（总结）:  LLM 生成推荐文字

入口: python app_v2.py
"""

import os
import json
import uuid
import time
import re
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI

from amap_client import (
    DEEPSEEK_API_KEY,
    AMAP_KEY,
    AMAP_JS_KEY,
    amap_geocode,
    amap_get_best_route,
    amap_search_nearby,
    haversine_distance,
    find_balanced_center,
)

app = Flask(__name__, static_folder="static")
CORS(app)

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

llm_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


# ──────────────────────────────────────────────────────
# Session 管理（内存缓存，TTL 1 小时）
# ──────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}
SESSION_TTL = 3600


def session_create(data: dict) -> str:
    sid = str(uuid.uuid4())[:8]
    _sessions[sid] = {"data": data, "expires_at": time.time() + SESSION_TTL}
    _session_cleanup()
    return sid


def session_get(sid: str) -> dict | None:
    s = _sessions.get(sid)
    if not s or s["expires_at"] < time.time():
        return None
    return s["data"]


def session_update(sid: str, data: dict) -> bool:
    if sid not in _sessions:
        return False
    _sessions[sid]["data"].update(data)
    _sessions[sid]["expires_at"] = time.time() + SESSION_TTL
    return True


def _session_cleanup():
    now = time.time()
    expired = [k for k, v in _sessions.items() if v["expires_at"] < now]
    for k in expired:
        del _sessions[k]


# ──────────────────────────────────────────────────────
# 通用工具函数
# ──────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | list | None:
    """从 LLM 回答中提取第一个合法 JSON 对象或数组"""
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _format_route(route: dict) -> dict:
    success = route.get("success", True)
    # 查询失败时保留 error 字段，duration_text 置为 None（前端据此显示友好提示）
    return {
        "mode":             route.get("mode", "unknown"),
        "success":          success,
        "error":            route.get("error") if not success else None,
        "duration_text":    route.get("duration_text") if success else None,
        "distance_text":    route.get("distance_text") if success else None,
        "line_summary":     route.get("line_summary", ""),
        "duration_minutes": route.get("duration_minutes", 999),
        "all_modes":        route.get("all_modes", {}),
    }


# ──────────────────────────────────────────────────────
# Agent 1：规划 Agent
# 职责：理解用户自然语言需求 → 结构化搜索参数
# 特点：无工具调用，上下文极短（~200 tokens in/out）
# ──────────────────────────────────────────────────────

_PLAN_SYSTEM = """\
你是搜索参数提取专家。根据用户的自然语言需求，提取结构化的搜索参数。
只输出一个JSON对象，不要有任何解释：

{
  "keyword": "高德地图搜索关键词（如：鲁菜、火锅、咖啡馆、电影院、KTV、酒吧）",
  "keyword_fallbacks": ["如果keyword搜索结果不足时的备选词1", "备选词2"],
  "min_rating": 4.0,
  "top_n": 20,
  "poi_category": "餐厅/咖啡馆/酒吧/电影院/KTV/购物/其他",
  "notes": "其他特殊需求，如价格范围、环境要求等，没有则空字符串",
  "sort_weights": {
    "rating": 0.6,
    "total_time": 0.3,
    "time_diff": 0.1
  }
}

规则：
- keyword 要精准（"鲁菜"而非"好吃的鲁菜餐厅"）
- min_rating：用户提到"评分高"→4.5，"评分4.5以上"→4.5，未提及→4.0
- top_n 固定为20（前端会提供数量选择器让用户筛选）
- keyword_fallbacks 提供2个备选，从窄到宽（例如"鲁菜"的备选：["山东菜","中餐"]）
- sort_weights 三项之和必须等于1.0，根据用户意图调整：
  * 默认（未明确提及）：rating=0.6, total_time=0.3, time_diff=0.1
  * 用户强调"近"/"不要太远"/"方便"：rating=0.3, total_time=0.6, time_diff=0.1
  * 用户强调"评分高"/"好评"/"口碑"：rating=0.8, total_time=0.15, time_diff=0.05
  * 用户强调"公平"/"两边一样远"/"均衡"：rating=0.4, total_time=0.2, time_diff=0.4
  * 用户同时强调近和公平：rating=0.2, total_time=0.4, time_diff=0.4
"""


def agent_plan(user_query: str) -> dict:
    """
    Agent 1：规划 Agent
    输入：用户需求文字
    输出：结构化搜索参数
    """
    print(f"[Agent1/规划] 分析需求: {user_query}")
    try:
        resp = llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user",   "content": user_query},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content or ""
        plan = _extract_json(content)
        if isinstance(plan, dict) and "keyword" in plan:
            print(f"[Agent1/规划] 结果: {plan}")
            return plan
    except Exception as e:
        print(f"[Agent1/规划] 错误: {e}")

    # 降级：直接用用户输入作为 keyword
    return {
        "keyword": user_query,
        "keyword_fallbacks": [],
        "min_rating": 4.0,
        "top_n": 20,
        "poi_category": "未知",
        "notes": "",
    }


# ──────────────────────────────────────────────────────
# Agent 2：搜索 Agent
# 职责：调用地图 API 搜索候选地点
# 先计算全员地理中心，再按 Agent1 提取的关键词搜索
# ──────────────────────────────────────────────────────


def agent_search(
    participants: list,
    plan: dict, city: str, search_ctx: dict,
) -> dict:
    """根据全员地理中心搜索候选地点。"""
    center_result = find_balanced_center(participants)
    midpoint = center_result["midpoint"]
    radius = center_result["suggested_search_radius_m"]
    keywords = [plan.get("keyword", "餐厅"), *plan.get("keyword_fallbacks", [])]

    for keyword in dict.fromkeys(filter(None, keywords)):
        result = amap_search_nearby(
            midpoint["lng"], midpoint["lat"], keyword, radius,
        )
        if result.get("pois"):
            search_ctx["pois"] = result["pois"]
            break

    return {
        "success": bool(search_ctx.get("pois")),
        "midpoint": midpoint,
        "search_radius_m": radius,
    }


# ──────────────────────────────────────────────────────
# 路线计算（纯 Python，不走 LLM）
# 职责：为 2-8 位参与者独立计算路线并聚合公平指标
# ──────────────────────────────────────────────────────

def _validate_participants(participants: list) -> list:
    """校验搜索边界上的参与者数组。"""
    if not isinstance(participants, list) or not 2 <= len(participants) <= 8:
        raise ValueError("参与者人数必须为 2-8 人")

    allowed_preferences = {"auto", "transit", "driving", "cycling", "walking"}
    ids = set()
    for index, participant in enumerate(participants, 1):
        if not isinstance(participant, dict):
            raise ValueError(f"参与者 {index} 数据格式错误")
        participant_id = participant.get("id")
        location = participant.get("location")
        if not participant_id or participant_id in ids:
            raise ValueError("参与者 ID 必须存在且唯一")
        ids.add(participant_id)
        if not isinstance(location, dict):
            raise ValueError(f"参与者 {index} 缺少地点")
        try:
            lng = float(location["lng"])
            lat = float(location["lat"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"参与者 {index} 坐标无效") from None
        if not (-180 <= lng <= 180 and -90 <= lat <= 90):
            raise ValueError(f"参与者 {index} 坐标无效")
        if participant.get("preference", "auto") not in allowed_preferences:
            raise ValueError(f"参与者 {index} 出行方式无效")
    return participants


def calculate_routes(
    pois: list,
    participants: list,
    city: str = "北京",
    departure_time: str | None = None,
    sort_weights: dict | None = None,
) -> list:
    """计算每个候选点的全员路线，并按评分、总耗时和时间极差排序。"""
    w = sort_weights or {}
    w_rating = float(w.get("rating", 0.6))
    w_total_time = float(w.get("total_time", 0.3))
    w_time_range = float(w.get("time_diff", w.get("time_range", 0.1)))
    w_sum = w_rating + w_total_time + w_time_range or 1.0
    w_rating /= w_sum
    w_total_time /= w_sum
    w_time_range /= w_sum

    enriched = []
    for poi in pois:
        routes = []
        for participant in participants:
            location = participant["location"]
            route = amap_get_best_route(
                location["lng"], location["lat"],
                poi["lng"], poi["lat"], city,
                participant.get("preference", "auto"), departure_time,
            )
            if not route.get("success", False):
                route = amap_get_best_route(
                    location["lng"], location["lat"],
                    poi["lng"], poi["lat"], city,
                    participant.get("preference", "auto"), departure_time,
                )
            if not route.get("success", False):
                routes = []
                break
            formatted = _format_route(route)
            formatted.update({
                "participant_id": participant["id"],
                "participant_name": participant.get("name") or participant["id"],
                "preference": participant.get("preference", "auto"),
            })
            routes.append(formatted)

        if not routes:
            continue

        durations = [route["duration_minutes"] for route in routes]
        enriched.append({
            **poi,
            "routes": routes,
            "total_time_minutes": sum(durations),
            "time_range_minutes": max(durations) - min(durations),
            "min_time_minutes": min(durations),
            "max_time_minutes": max(durations),
        })

    max_total = max((p["total_time_minutes"] for p in enriched), default=1) or 1
    max_range = max((p["time_range_minutes"] for p in enriched), default=1) or 1
    max_rating = max((p.get("rating", 0) for p in enriched), default=5) or 5
    for poi in enriched:
        poi["_score"] = round(
            (poi.get("rating", 0) / max_rating) * w_rating
            - (poi["total_time_minutes"] / max_total) * w_total_time
            - (poi["time_range_minutes"] / max_range) * w_time_range,
            4,
        )

    enriched.sort(key=lambda item: item["_score"], reverse=True)
    return enriched


def filter_and_rank_pois(
    pois: list, min_rating: float = 4.0, top_n: int = 20
) -> list:
    """筛选评分 >= min_rating，按评分降序，取前 top_n 个（最多 20）"""
    top_n = min(top_n, 20)
    filtered = [p for p in pois if p.get("rating", 0) >= min_rating]
    filtered.sort(key=lambda x: x.get("rating", 0), reverse=True)
    if len(filtered) < 3 and pois:
        filtered = sorted(pois, key=lambda x: x.get("rating", 0), reverse=True)
    return filtered[:top_n]


# ──────────────────────────────────────────────────────
# Agent 3：总结 Agent
# 职责：根据结构化数据生成自然语言推荐文字
# 特点：无工具，上下文可控（只传摘要）
# ──────────────────────────────────────────────────────

_SUMMARY_SYSTEM = """\
你是一个简洁友好的推荐助手。根据给出的地点数据，用2-4句话概括推荐结果。
重点提炼：
- 在哪个区域找到了什么类型的地点
- 评分最高的是哪家，全员交通是否均衡
- 如果有地方最快与最慢用时差距大，提醒一下
语气活泼简洁，不要罗列所有细节，不要用"首先其次"等套话。
"""


def agent_summarize(
    query: str,
    participants: list,
    enriched_pois: list,
) -> str:
    """Agent 3：用全员路线摘要生成简短推荐文字。"""
    print("[Agent3/总结] 生成推荐文字...")
    mode_label = {
        "auto": "最快方式", "transit": "公交地铁",
        "driving": "驾车", "cycling": "骑行", "walking": "步行",
    }

    pois_summary = []
    for index, poi in enumerate(enriched_pois[:5]):
        route_text = " / ".join(
            f"{route['participant_name']} {mode_label.get(route.get('mode'), route.get('mode'))} {route.get('duration_text', '?')}"
            for route in poi.get("routes", [])
        )
        pois_summary.append(
            f"{index + 1}. {poi['name']}（评分{poi.get('rating', 0):.1f}）"
            f" - {route_text}，最大时间差{poi.get('time_range_minutes', '?')}分钟"
        )

    participant_text = "\n".join(
        f"{p.get('name') or p['id']}（{p['location'].get('name', '未命名地点')}）"
        f"出行方式：{mode_label.get(p.get('preference', 'auto'), p.get('preference', 'auto'))}"
        for p in participants
    )
    user_msg = f"用户需求：{query}\n参与者：\n{participant_text}\n找到地点：\n" + "\n".join(pois_summary)

    try:
        resp = llm_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or "已找到推荐地点，请查看地图标注。"
    except Exception as e:
        print(f"[Agent3/总结] 错误: {e}")
        return f"已找到 {len(enriched_pois)} 个推荐地点。"


# ──────────────────────────────────────────────────────
# 完整流水线（供 /api/v2/search 调用）
# ──────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    participants: list,
    city: str = "北京",
    departure_time: str | None = None,
) -> dict:
    """运行 2-8 人智能中间点推荐流水线。"""
    plan = agent_plan(user_query)

    search_ctx: dict = {"pois": []}
    search_result = agent_search(participants, plan, city, search_ctx)
    if not search_result.get("success"):
        return {"success": False, "error": "未能找到符合条件的地点，请尝试修改关键词或扩大范围"}

    midpoint = search_result["midpoint"]
    search_radius_m = search_result["search_radius_m"]
    top_pois = filter_and_rank_pois(
        search_ctx["pois"],
        min_rating=plan.get("min_rating", 4.0),
        top_n=min(plan.get("top_n", 12), 12),
    )
    enriched = calculate_routes(
        top_pois, participants, city, departure_time,
        sort_weights=plan.get("sort_weights"),
    )
    if not enriched:
        return {"success": False, "error": "部分参与者路线无法计算，请调整地点或交通方式"}

    summary_text = agent_summarize(user_query, participants, enriched)
    session_id = session_create({
        "participants": participants,
        "city": city,
        "query": user_query,
        "plan": plan,
        "midpoint": midpoint,
        "search_radius_m": search_radius_m,
        "pois_base": top_pois,
        "departure_time": departure_time,
    })
    return {
        "success": True,
        "session_id": session_id,
        "summary": summary_text,
        "plan": plan,
        "midpoint": midpoint,
        "search_radius_m": search_radius_m,
        "pois": enriched,
        "participants": participants,
    }


# ──────────────────────────────────────────────────────
# Flask 路由
# ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/config")
def get_config():
    return jsonify({
        "amap_key": AMAP_JS_KEY,
        "has_amap_key": bool(AMAP_JS_KEY),
        "version": "v2",
    })


@app.route("/api/v2/search", methods=["POST"])
def api_v2_search():
    """主搜索接口：接收 2-8 位参与者并运行推荐流水线。"""
    data = request.json or {}
    user_query = data.get("query", "").strip()
    city = data.get("city", "北京")
    departure_time = data.get("departure_time") or None

    try:
        participants = _validate_participants(data.get("participants"))
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    if not user_query:
        return jsonify({"success": False, "error": "请描述您的需求"}), 400
    if not AMAP_KEY:
        return jsonify({"success": False, "error": "高德地图 API Key 未配置"}), 500
    if not DEEPSEEK_API_KEY:
        return jsonify({"success": False, "error": "DeepSeek API Key 未配置"}), 500

    try:
        result = run_pipeline(user_query, participants, city, departure_time)
        return jsonify(result), (200 if result.get("success") else 500)
    except Exception as error:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/api/v2/routes", methods=["POST"])
def api_v2_routes():
    """复用候选地点，按最新全员偏好重算路线。"""
    data = request.json or {}
    session_id = data.get("session_id")
    departure_time = data.get("departure_time") or None
    if not session_id:
        return jsonify({"success": False, "error": "缺少 session_id，请先执行搜索"}), 400

    session = session_get(session_id)
    if not session:
        return jsonify({"success": False, "error": "会话已过期，请重新搜索"}), 404
    try:
        participants = _validate_participants(data.get("participants", session["participants"]))
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 400
    if departure_time is None:
        departure_time = session.get("departure_time")

    try:
        enriched = calculate_routes(
            session.get("pois_base", []), participants,
            session.get("city", "北京"), departure_time,
            sort_weights=session.get("plan", {}).get("sort_weights"),
        )
        if not enriched:
            return jsonify({"success": False, "error": "部分参与者路线无法计算，请调整地点或交通方式"}), 500
        session_update(session_id, {
            "participants": participants,
            "departure_time": departure_time,
        })
        return jsonify({
            "success": True,
            "session_id": session_id,
            "pois": enriched,
            "participants": participants,
            "midpoint": session.get("midpoint", {}),
            "search_radius_m": session.get("search_radius_m", 3000),
        })
    except Exception as error:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(error)}), 500


@app.route("/api/v2/session/<session_id>", methods=["GET"])
def api_v2_session_info(session_id):
    """查看 Session 摘要（调试用）。"""
    session = session_get(session_id)
    if not session:
        return jsonify({"exists": False}), 404
    return jsonify({
        "exists": True,
        "query": session.get("query"),
        "city": session.get("city"),
        "poi_count": len(session.get("pois_base", [])),
        "participant_count": len(session.get("participants", [])),
        "participants": session.get("participants", []),
        "plan": session.get("plan"),
    })


@app.route("/api/geocode", methods=["POST"])
def api_geocode():
    data = request.json or {}
    address = data.get("address", "")
    if not address:
        return jsonify({"success": False, "error": "请提供地址"}), 400
    return jsonify(amap_geocode(address))


@app.route("/api/nearby-search", methods=["POST"])
def api_nearby_search():
    """
    查询某坐标附近的 POI（用于卡片内"附近搜索"功能）。
    请求：{ lng, lat, keyword, radius_m (可选，默认 1000) }
    返回：{ success, pois: [{name, address, distance_m, rating, lng, lat, type}] }
    """
    data     = request.json or {}
    lng      = data.get("lng")
    lat      = data.get("lat")
    keyword  = data.get("keyword", "").strip()
    radius_m = min(int(data.get("radius_m", 1000)), 5000)

    if not lng or not lat or not keyword:
        return jsonify({"success": False, "error": "缺少参数 lng/lat/keyword"}), 400

    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "key":       AMAP_KEY,
        "location":  f"{lng},{lat}",
        "keywords":  keyword,
        "radius":    radius_m,
        "offset":    10,
        "page":      1,
        "extensions": "base",
        "output":    "json",
    }
    try:
        resp   = requests.get(url, params=params, timeout=8)
        result = resp.json()
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "pois": []})

    if result.get("status") != "1":
        return jsonify({"success": False, "error": result.get("info", "搜索失败"), "pois": []})

    pois = []
    for p in result.get("pois", []):
        loc = p.get("location", "").split(",")
        if len(loc) != 2:
            continue
        try:
            plng, plat = float(loc[0]), float(loc[1])
            dist       = int(haversine_distance(lng, lat, plng, plat) * 1000)
            biz_ext    = p.get("biz_ext") or {}
            rating     = float(biz_ext.get("rating", 0) or 0) if isinstance(biz_ext, dict) else 0.0
            pois.append({
                "name":       p.get("name", ""),
                "address":    p.get("address", "") if isinstance(p.get("address"), str) else "",
                "type":       p.get("type", ""),
                "distance_m": dist,
                "rating":     rating,
                "lng":        plng,
                "lat":        plat,
            })
        except (ValueError, TypeError):
            continue

    pois.sort(key=lambda x: x["distance_m"])
    return jsonify({"success": True, "pois": pois[:10]})


@app.route("/api/geocode-suggest", methods=["POST"])
def api_geocode_suggest():
    """
    地点输入提示接口 — 前端搜索框下拉候选。
    调用高德 /v3/assistant/inputtips，返回带坐标的候选列表。
    """
    data    = request.json or {}
    keyword = data.get("keyword", "").strip()
    if not keyword:
        return jsonify({"tips": []})

    url = "https://restapi.amap.com/v3/assistant/inputtips"
    params = {
        "key":      AMAP_KEY,
        "keywords": keyword,
        "datatype": "all",
        "output":   "json",
    }
    try:
        resp   = requests.get(url, params=params, timeout=8)
        result = resp.json()
    except Exception as e:
        return jsonify({"tips": [], "error": str(e)})

    tips = []
    if result.get("status") == "1":
        for tip in result.get("tips", []):
            location = tip.get("location", "")
            if not location or location == "[]":
                continue
            try:
                lng_s, lat_s = location.split(",")
                tips.append({
                    "name":     tip.get("name", ""),
                    "district": tip.get("district", ""),
                    "address":  tip.get("address", "") if isinstance(tip.get("address"), str) else "",
                    "lng":      float(lng_s),
                    "lat":      float(lat_s),
                })
            except (ValueError, TypeError):
                continue

    return jsonify({"tips": tips[:6]})


if __name__ == "__main__":
    print("=" * 60)
    print("  智能中间点推荐系统 v2（多 Agent 架构）")
    print("=" * 60)
    print(f"  DeepSeek API Key: {'已配置' if DEEPSEEK_API_KEY else '未配置 ⚠️'}")
    print(f"  高德地图 API Key:  {'已配置' if AMAP_KEY else '未配置 ⚠️'}")
    print(f"  Session 缓存: 内存（TTL {SESSION_TTL // 3600} 小时）")
    print("=" * 60)
    print("  访问地址: http://localhost:5000")
    print("=" * 60)
    # threaded=True：每个请求在独立线程中处理，允许多用户同时访问/打开多个网页
    # 不加此参数（或设为 False）时，Flask 单线程串行处理，一个请求卡住会阻塞所有人
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
