"""
vlm_prompts.py
Системный промпт и разбор ответа локальной VLM (Qwen2-VL).
VLM получает пронумерованные кадры окна вокруг CV-триггера и обязана вернуть строгий JSON:
есть ли видимое нарушение, его код из фиксированного списка, номер кадра-доказательства и
наблюдение без домыслов. Код напрямую сопоставляется со статьей в legal_base.LEGAL_ARTICLES_RK.
"""

import json
import re

# Код -> визуальный признак, по которому VLM узнает нарушение.
# Ключи совпадают с ключами legal_base.LEGAL_ARTICLES_RK.
VIOLATION_TYPES = {
    "LEAVE_WORKPLACE": "кассир встает и уходит от стола, а на столе остаются деньги, документы, печать, мешок или включенный ноутбук",
    "UNLOCKED_SAFE": "дверца сейфа открыта или в замке сейфа торчит ключ, когда кассир отходит или рядом другие люди",
    "NO_ID_CHECK": "кассир проводит операцию с клиентом, но клиент не передает удостоверение личности и кассир его не смотрит",
    "CASH_NOT_COUNTED": "кассир принимает деньги (в том числе из пакета или сумки) и убирает их без полистного пересчета",
    "COUNTING_AWAY_FROM_CLIENT": "кассир пересчитывает деньги, когда клиента уже нет у кассы",
    "INCOMPLETE_PAYOUT": "кассир отдает клиенту не все подготовленные деньги, часть остается у кассира",
    "MIXED_CASH": "деньги прошлого клиента остаются на столе при следующем клиенте, или деньги разных клиентов складываются в одну пачку",
    "SUSPICIOUS_NOTES_MIXED": "купюры, отбракованные счетной машинкой, кассир кладет в общую пачку с остальными",
    "THIRD_PERSON_INTRUSION": "посторонний человек (не кассир и не обслуживаемый клиент) заходит в кассу или заглядывает в дверь",
    "DOOR_OPEN": "дверь кассы остается открытой после того, как кто-то вошел или вышел",
    "FOOD_DRINK_AT_DESK": "кассир пьет или ест на рабочем месте",
    "PERSONAL_ITEMS": "на рабочем столе кассира лежат личные вещи (сумка, косметика, еда, телефон)",
    "CLIENT_DESK_ACCESS": "клиент сам тянется в рабочую зону кассира или берет деньги со стола кассира",
    "OTHER": "другое явно видимое нарушение порядка работы с наличными, не подходящее под коды выше",
}

SYSTEM_PROMPT = (
    "Ты помогаешь контролеру банка проверять видеозаписи кассы. "
    "Тебе показывают пронумерованные кадры с потолочной камеры: кассир за столом, деньги, сейф, ноутбук, "
    "иногда клиент, инкассатор или посторонний.\n"
    "Задача: решить, видно ли на кадрах нарушение из списка ниже.\n\n"
    "Правила:\n"
    "1. Опирайся только на то, что видно на кадрах. Ничего не додумывай.\n"
    "2. Если на кадрах обычная работа, ответь violation=false. Само наличие денег, сейфа или клиента нарушением не является.\n"
    "3. Не делай выводов о том, что нельзя увидеть: суммы, проводки, данные на экране, содержание документов.\n"
    "4. Не описывай внешность и не узнавай людей. Называй их по роли: кассир, клиент, инкассатор, посторонний.\n"
    "5. type выбирай строго из списка кодов. Если нарушение видно, но код не подходит, используй OTHER.\n"
    "6. frame - номер кадра, на котором нарушение видно лучше всего.\n"
    "7. observation - одно-два коротких предложения по-русски: что именно видно и почему это нарушение.\n"
    "8. Подсказка детектора движения может быть ложной: проверяй ее по кадрам.\n\n"
    "Коды нарушений:\n"
    + "\n".join(f"- {code}: {cue}" for code, cue in VIOLATION_TYPES.items())
    + "\n\nОтветь ТОЛЬКО одним JSON-объектом, без текста до и после:\n"
    '{"violation": true или false, "type": "КОД" или null, "frame": номер кадра, "observation": "..."}'
)


def build_messages(frames, start_t, end_t, event_hint=None):
    """
    frames: список (timestamp_sec, PIL.Image).
    Кадры подаются отдельными изображениями с подписью номера и таймкода,
    чтобы VLM могла указать конкретный кадр-доказательство.
    """
    content = [{"type": "text", "text": f"Кадры с {start_t:.1f} по {end_t:.1f} с."}]
    for i, (t, img) in enumerate(frames, start=1):
        content.append({"type": "text", "text": f"\nКадр {i} ({t:.1f} с):"})
        content.append({"type": "image", "image": img})
    if event_hint:
        content.append({"type": "text", "text": f"\nПодсказка детектора движения: {event_hint}"})
    content.append({"type": "text", "text": "\nОтветь JSON."})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def parse_vlm_response(text, n_frames):
    """
    Достает JSON из ответа VLM. Возвращает dict с ключами violation/type/frame/observation
    или None, если ответ разобрать не удалось.
    """
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    raw = m.group(0)
    data = None
    for candidate in (raw, raw.replace("True", "true").replace("False", "false").replace("None", "null").replace("'", '"')):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(data, dict):
        return None

    violation = data.get("violation")
    if isinstance(violation, str):
        violation = violation.strip().lower() in ("true", "да", "yes")
    code = str(data.get("type") or "").strip().upper()
    if code not in VIOLATION_TYPES:
        code = "OTHER" if violation else None
    try:
        frame = int(data.get("frame") or 0)
    except (TypeError, ValueError):
        frame = 0
    frame = min(max(frame, 1), n_frames) if n_frames else 0

    return {
        "violation": bool(violation),
        "type": code,
        "frame": frame,
        "observation": str(data.get("observation") or "").strip(),
    }
