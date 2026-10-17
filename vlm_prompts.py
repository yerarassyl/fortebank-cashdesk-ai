"""
vlm_prompts.py
Проверка окон CV-триггеров локальной VLM через атомарные визуальные вопросы.

Схема:
1. Каждому CV-триггеру соответствует чек-лист (из prompt_focus в cv_triggers.py), разложенный
   на короткие вопросы "Да/Нет" об одном видимом факте. Плюс базовые вопросы на каждом окне.
2. VLM отвечает на все вопросы по нескольким кадрам окна; берется вероятность ответа "Да"
   (а не сгенерированный текст) — маленькая модель так отвечает стабильнее и порог настраивается.
3. Решение принимает код по правилам: какой факт, в каком контексте и на скольких кадрах
   означает нарушение. VLM не "решает", что это нарушение, она только сообщает, что видно.
"""

# id -> вопрос. Для фактов-нарушений "Да" = тревожный признак.
QUESTIONS = {
    # контекст сцены
    "cashier_at_desk": "Кассир сидит за деревянным столом?",
    "client_present": "Перед столом кассира стоит другой человек?",
    "extra_person": "В кабинете находятся три человека или больше?",
    # базовые факты
    "money_desk": "На столе лежат деньги или пачки купюр?",
    "docs_desk": "На столе лежат документы, печать или мешок?",
    "safe_open": "Дверца серого металлического сейфа открыта?",
    "door_open": "Дверь в кабинет открыта?",
    "personal_items": "На столе лежат личные вещи: сумка, косметичка, еда или напиток?",
    # чек-листы триггеров
    "key_in_safe": "В замке сейфа торчит ключ?",
    "laptop_on": "Ноутбук на столе открыт и его экран включен?",
    "id_shown": "Клиент показывает или передает кассиру удостоверение личности или карточку?",
    "counting": "Кассир пересчитывает деньги в руках?",
    "peeking": "Кто-то заглядывает в кабинет из дверного проема?",
    "drinking": "Кассир пьет из чашки, стакана или бутылки?",
    "eating": "Кассир ест?",
    "hand_to_hand": "Деньги передаются из рук в руки, минуя стол?",
    "machine_used": "Кассир кладет купюры в счетную машинку?",
}

QUESTION_SUFFIX = " Ответь Да или Нет."

# Базовые вопросы — на каждом окне независимо от триггера
FUNDAMENTAL = ["cashier_at_desk", "client_present", "extra_person", "money_desk", "docs_desk",
               "safe_open", "door_open", "personal_items"]

# Чек-листы CV-триггеров (prompt_focus в cv_triggers.py), разложенные на атомарные вопросы
TRIGGER_QUESTIONS = {
    # "Закрыта ли дверца сейфа, заперт ли замок? Нет ли на столе посторонних предметов, денег, документов? Закрыта ли дверь?"
    "INITIAL_DESK_CHECK": ["key_in_safe"],
    # "Заблокирован ли ноутбук? Заперт ли сейф / ключ в замке? Убраны ли деньги и документы? Закрыта ли дверь?"
    "CASHIER_STANDUP": ["laptop_on", "key_in_safe"],
    # "Проверила ли удостоверение? Остались ли деньги от прошлой операции? Закрыта ли дверь?"
    "CLIENT_ARRIVAL": ["id_shown"],
    # "Убрала ли деньги в сейф? Пересчитывает ли деньги после ухода клиента? Закрыта ли дверь?"
    "CLIENT_DEPARTURE": ["counting"],
    # "Зашел ли посторонний внутрь? Подсматривает ли из дверного проема?"
    "THIRD_PERSON": ["peeking"],
    # "Пьет ли кассир воду/напиток или принимает пищу?"
    "HAND_TO_MOUTH": ["drinking", "eating"],
    # "Вернула ли деньги через лоток? Проверены ли на машинке или из рук в руки?"
    "HANDOVER": ["id_shown", "hand_to_hand", "machine_used", "counting"],
}

# Пороги: вероятность "Да" и минимальное число кадров, на которых факт держится.
# Начальные значения — подлежат калибровке на обучающем датасете.
THRESHOLD = 0.5
ABSENT_THRESHOLD = 0.15  # для нарушений "по бездействию": факт ни разу не увиден


def questions_for(triggers):
    """Набор вопросов для окна: базовые + чек-листы всех триггеров, слитых в окно."""
    ids = list(FUNDAMENTAL)
    for trig in triggers:
        for q in TRIGGER_QUESTIONS.get(trig, []):
            if q not in ids:
                ids.append(q)
    return ids


def _frames_where(values, thr):
    return [i for i, v in enumerate(values) if v >= thr]


def _rule(code, per_frame, min_frames, observation):
    """per_frame — комбинированная вероятность по кадрам. Нарушение, если держится на min_frames кадрах."""
    hits = _frames_where(per_frame, THRESHOLD)
    if len(hits) < min_frames:
        return None
    best = max(hits, key=lambda i: per_frame[i])
    return {"code": code, "frame": best, "score": per_frame[best], "observation": observation}


def decide(scores, triggers):
    """
    scores: {question_id: [P(Да) по кадрам]}; triggers: список CV-триггеров окна.
    Возвращает список найденных нарушений {code, frame, score, observation}.
    """
    n = len(next(iter(scores.values())))
    get = lambda q: scores.get(q, [0.0] * n)
    absent = [1 - v for v in get("cashier_at_desk")]
    others = [max(c, e) for c, e in zip(get("client_present"), get("extra_person"))]
    trig = set(triggers)
    found = []

    # Ценности на столе, а кассира за столом нет (НБРК № 231, п. 20, 22)
    found.append(_rule("LEAVE_WORKPLACE",
                       [min(max(m, d), a) for m, d, a in zip(get("money_desk"), get("docs_desk"), absent)], 2,
                       "Кассира нет за столом, а на столе остаются деньги, документы, печать или мешок."))
    # Сейф открыт или ключ в замке, когда кассир отошел или рядом другие люди (п. 20, 80)
    found.append(_rule("UNLOCKED_SAFE",
                       [min(max(s, k), max(a, o)) for s, k, a, o in zip(get("safe_open"), get("key_in_safe"), absent, others)], 2,
                       "Дверца сейфа открыта или в замке ключ, когда кассир отошел или рядом другие люди."))
    # Ноутбук включен, кассира нет (АРРФР № 53, п. 71)
    if "CASHIER_STANDUP" in trig:
        found.append(_rule("SCREEN_UNLOCKED", [min(l, a) for l, a in zip(get("laptop_on"), absent)], 2,
                           "Кассир отошла от стола, а ноутбук остается открытым и включенным."))
    # Клиент пришел, а на столе деньги прошлой операции (п. 20, 21): смотрим первые кадры окна
    if "CLIENT_ARRIVAL" in trig:
        head = max(1, n // 3)
        found.append(_rule("MIXED_CASH", [min(m, c) for m, c in zip(get("money_desk")[:head], get("client_present")[:head])], 1,
                           "К кассе подошел клиент, а на столе еще лежат деньги."))
    # Клиент у кассы, но удостоверение ни разу не показано (п. 19, 31) — нарушение по бездействию
    if trig & {"CLIENT_ARRIVAL", "HANDOVER"}:
        client_frames = _frames_where(get("client_present"), THRESHOLD)
        if len(client_frames) >= 2 and max(get("id_shown")[i] for i in client_frames) < ABSENT_THRESHOLD:
            found.append({"code": "NO_ID_CHECK", "frame": client_frames[0], "score": 1 - max(get("id_shown")),
                          "observation": "Клиент находится у кассы, но на кадрах не видно, чтобы он показывал удостоверение личности."})
    # Пересчет денег без клиента (п. 20, 31)
    if "CLIENT_DEPARTURE" in trig:
        found.append(_rule("COUNTING_AWAY_FROM_CLIENT",
                           [min(c, 1 - p) for c, p in zip(get("counting"), get("client_present"))], 2,
                           "Кассир пересчитывает деньги, когда клиента у кассы уже нет."))
    # Деньги из рук в руки без счетной машинки (п. 9, 19)
    if "HANDOVER" in trig:
        found.append(_rule("NO_COUNTING_MACHINE",
                           [min(h, 1 - m) for h, m in zip(get("hand_to_hand"), get("machine_used"))], 2,
                           "Деньги передаются из рук в руки, без проверки на счетной машинке."))
    # Посторонний внутри или заглядывает (№ 14 п. 8)
    found.append(_rule("THIRD_PERSON_INTRUSION", [max(e, p) for e, p in zip(get("extra_person"), get("peeking"))], 2,
                       "В кабинете находится посторонний человек или кто-то заглядывает в дверь."))
    # Дверь остается открытой в конце окна (№ 14 п. 16)
    tail = n - max(1, n // 3)
    found.append(_rule("DOOR_OPEN", [0.0] * tail + get("door_open")[tail:], 2,
                       "Дверь в кабинет остается открытой."))
    # Еда/питье (п. 12) — короткое действие, достаточно одного кадра
    found.append(_rule("FOOD_DRINK_AT_DESK", [max(d, e) for d, e in zip(get("drinking"), get("eating"))], 1,
                       "Кассир пьет или ест на рабочем месте."))
    # Личные вещи на столе (п. 12)
    found.append(_rule("PERSONAL_ITEMS", get("personal_items"), 2,
                       "На рабочем столе кассира лежат личные вещи."))

    return [f for f in found if f]
