import os, sys, re, pandas as pd
sys.path.insert(0, r'e:\forte-cashdesk-ai')
from cv_triggers import PoseEventDetector

def parse_target_sec(ts_raw):
    if pd.isna(ts_raw): return None
    s = str(ts_raw).lower().strip()
    m_min = re.search(r'(\d+)\s*мин', s)
    m_sec = re.search(r'(\d+)\s*сек', s)
    minutes = int(m_min.group(1)) if m_min else 0
    seconds = int(m_sec.group(1)) if m_sec else 0
    if not m_min and not m_sec:
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if m: return float(m.group(1))
        return None
    return float(minutes * 60 + seconds)

detector = PoseEventDetector()
csv_path = r'e:\open\open_labels_copy.csv'
df = pd.read_csv(csv_path)
video_files = os.listdir(r'e:\open\videos')
def norm(s): return str(s).strip().replace('Х','X').replace('х','x').lower()
fmap = {norm(os.path.splitext(f)[0]): os.path.join(r'e:\open\videos', f) for f in video_files}

rows = []
trigger_counts = {}
for idx, r in df.iterrows():
    name = str(r.iloc[0]).strip()
    status = str(r.iloc[2]).strip()
    is_violation = (status.lower() == 'нарушение')
    t_sec = parse_target_sec(r.iloc[1]) if is_violation else None
    vpath = fmap.get(norm(name))
    if not vpath or not os.path.exists(vpath): continue
    evs = detector.detect_video_events(vpath)
    for e in evs:
        t_type = e['trigger']
        trigger_counts[t_type] = trigger_counts.get(t_type, 0) + 1
    min_d = min([abs(e['timecode_sec'] - t_sec) for e in evs]) if (is_violation and evs) else 999.0
    best = None
    if is_violation and evs:
        be = min(evs, key=lambda e: abs(e['timecode_sec'] - t_sec))
        best = f"{be['timecode_sec']}s [{be['trigger']}]"
    rows.append({
        'Video': name,
        'Status': status,
        'Triggers': len(evs),
        'Caught': (min_d <= 10.0) if is_violation else True,
        'MinDelta': round(min_d, 2) if is_violation else None,
        'Best': best
    })

rep = pd.DataFrame(rows)
rep.to_csv(r'e:\forte-cashdesk-ai\all_42_temporal_audit.csv', index=False, encoding='utf-8-sig')
print('=== BREAKDOWN OF 106 TRIGGERS ===')
for k, v in sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True):
    print(f'   {k:<22}: {v}')
