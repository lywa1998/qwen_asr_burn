#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path('/Users/riddler/Github/qwen_asr_burn')
ALIGN = ROOT / 'test_30s_srt_work' / 'segment_000.align.json'
SEG_OFFSET = 0.08
OUT_JSON = ROOT / 'test_30s_srt_work' / 'phrases.json'
OUT_SOURCE_SRT = ROOT / 'test_30s.aligned.srt'

PUNCT_STRONG = set('。！？.!?;；')
PUNCT_COMMA = set('，,、')
MAX_DUR = 4.0
MIN_COMMA_DUR = 1.6
MAX_CHARS = 24

items = json.loads(ALIGN.read_text())
phrases = []
cur = []

def token_text(t):
    return (t.get('text') or '').strip()

def flush():
    global cur
    if not cur:
        return
    text = ''.join(token_text(x) for x in cur).strip()
    if not text:
        cur = []
        return
    start = min(float(x['start_time']) for x in cur if token_text(x)) + SEG_OFFSET
    end = max(float(x['end_time']) for x in cur if token_text(x)) + SEG_OFFSET
    if end <= start:
        end = start + 0.2
    phrases.append({'start': start, 'end': end, 'text': text})
    cur = []

for item in items:
    txt = token_text(item)
    if not txt:
        continue
    cur.append(item)
    text = ''.join(token_text(x) for x in cur)
    start = float(cur[0]['start_time'])
    end = max(float(x['end_time']) for x in cur)
    dur = end - start
    break_now = False
    if txt in PUNCT_STRONG:
        break_now = True
    elif txt in PUNCT_COMMA and (dur >= MIN_COMMA_DUR or len(text) >= 16):
        break_now = True
    elif dur >= MAX_DUR or len(text) >= MAX_CHARS:
        break_now = True
    if break_now:
        flush()
flush()

# Merge very short punctuation-only tails into previous if any were created.
merged = []
for p in phrases:
    if merged and len(p['text']) <= 2 and re.fullmatch(r'[。！？.!?;；，,、]+', p['text']):
        merged[-1]['text'] += p['text']
        merged[-1]['end'] = p['end']
    else:
        merged.append(p)
phrases = merged

def fmt(t):
    if t < 0:
        t = 0
    ms_total = int(round(t * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

OUT_JSON.write_text(json.dumps(phrases, ensure_ascii=False, indent=2))
with OUT_SOURCE_SRT.open('w') as f:
    for i, p in enumerate(phrases, 1):
        f.write(f'{i}\n{fmt(p["start"])} --> {fmt(p["end"])}\n{p["text"]}\n\n')
print(f'wrote {len(phrases)} phrases')
for i, p in enumerate(phrases, 1):
    print(i, fmt(p['start']), '-->', fmt(p['end']), p['text'])
