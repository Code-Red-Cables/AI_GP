import json, sys, statistics as st, collections

path = sys.argv[1] if len(sys.argv) > 1 else 'logs/latest.jsonl'
rows = [json.loads(l) for l in open(path) if l.strip()]
meta = rows[0]
rows = rows[1:]
arm = [r for r in rows if r.get('armed')]
print("file=%s rows=%d arm=%d" % (path, len(rows), len(arm)))
# sample a few vis fields raw
shown = 0
for r in arm:
    if r.get('vis'):
        print("sample vis:", r['vis'])
        shown += 1
        if shown >= 3:
            break

bym = collections.defaultdict(list)
det_count = 0
for r in arm:
    v = r.get('vis')
    if v and v[0]:
        det_count += 1
        if v[2] is not None:
            bym[v[3]].append(v[2])
print("detected rows:", det_count)
for m, vals in bym.items():
    print("  method=%s n=%d min=%.1f median=%.1f max=%.1f" % (
        m, len(vals), min(vals), st.median(vals), max(vals)))
