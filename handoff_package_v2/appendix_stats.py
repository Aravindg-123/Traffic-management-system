import pandas as pd

probs = pd.read_csv('transition_probs_1st.csv', index_col=0)
counts = pd.read_csv('transition_counts.csv', index_col=0)

rows = []
for f in probs.index:
    for t in probs.columns:
        c = counts.loc[f, t]
        p = probs.loc[f, t]
        if c > 0:
            rows.append((f, t, p, c))

df = pd.DataFrame(rows, columns=['from', 'to', 'prob', 'count'])

print('--- TOP 10 MOST PREDICTABLE (highest prob) ---')
top10 = df.sort_values('prob', ascending=False).head(10)
for _, r in top10.iterrows():
    print(f"{r['from']} -> {r['to']}: {r['prob']:.4f} (count={int(r['count'])})")

print()
print('--- TOP 10 LEAST PREDICTABLE (lowest prob, among observed nonzero) ---')
bot10 = df.sort_values('prob', ascending=True).head(10)
for _, r in bot10.iterrows():
    print(f"{r['from']} -> {r['to']}: {r['prob']:.4f} (count={int(r['count'])})")

print()
print('Count with prob > 0.60:', (df['prob'] > 0.60).sum())
print('Count with prob < 0.15:', (df['prob'] < 0.15).sum())
