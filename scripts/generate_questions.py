#!/usr/bin/env python3
"""Synthetic physician-style questions for the retrieval test (ONLY as a stand-in until physicians are available).

For every seed passage (results/annotation/seed_passages.csv) a GENERATOR model writes up to two questions:
  * type 'recommendation' - answered by the recommendation text;
  * type 'evidence'       - asks for the strength/level of evidence of that recommendation (only if the passage has one).
A different VALIDATOR model then checks each question (self-contained? answered by the passage? not a copy of its wording?).
Only questions that pass go to the JSONL (format of src/ascm/retrieval.py). Generator never sees chunk boundaries or any method.
Needs: pip install anthropic ; export ANTHROPIC_API_KEY=... ; models given by --gen-model / --val-model.
"""
import argparse, csv, json, os, re, sys

GEN = """Ты врач-онколог/хирург и ищешь в клинических рекомендациях ответ на вопрос из практики.
Язык вопроса: {lang}. Фрагмент рекомендаций (документ: {doc}):

{passage}
{evidence_line}
Сформулируй вопросы так, как их задал бы врач поисковой системе, НЕ видя фрагмента:
- вопрос должен быть понятен сам по себе (назови заболевание/ситуацию, не пиши «в этом фрагменте», «выше»);
- не копируй формулировки фрагмента дословно, перефразируй;
- вопрос 1 (type=recommendation): клинический вопрос, ответом на который служит рекомендация из фрагмента;
{q2}Верни ТОЛЬКО JSON: {{"questions":[{{"type":"recommendation","question":"..."}}{q2json}]}}"""
Q2 = "- вопрос 2 (type=evidence): спроси, насколько сильна рекомендация / каков уровень доказательности именно для этого утверждения;\n"
VAL = """Фрагмент:
{passage}
{evidence}
Вопрос: {question}
Оцени (true/false) и верни ТОЛЬКО JSON {{"self_contained":..,"answered_by_passage":..,"not_copied":..}}:
self_contained - вопрос понятен без фрагмента и не содержит отсылок к нему;
answered_by_passage - фрагмент (для type=evidence вместе со строкой доказательности) отвечает на вопрос;
not_copied - вопрос не повторяет дословно длинные куски фрагмента."""

def ask(client, model, prompt, n=600):
    r = client.messages.create(model=model, max_tokens=n, messages=[{"role": "user", "content": prompt}])
    t = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text')
    m = re.search(r'\{.*\}', t, re.S)
    return json.loads(m.group(0)) if m else {}

def run(client, rows, gen_model, val_model, lang_of, log=print):
    out, dropped = [], 0
    for r in rows:
        has_ev = bool(r['evidence'])
        ev_line = f"\nСтрока уровня доказательности: {r['evidence']}\n" if has_ev else ''
        p = GEN.format(lang=lang_of(r['doc']), doc=r['doc'], passage=r['passage'], evidence_line=ev_line,
                       q2=Q2 if has_ev else '', q2json=',{"type":"evidence","question":"..."}' if has_ev else '')
        try:
            qs = ask(client, gen_model, p).get('questions', [])
        except Exception as e:
            log('gen fail', r['id'], e); continue
        for k, q in enumerate(qs):
            typ, text = q.get('type', ''), (q.get('question') or '').strip()
            if typ not in ('recommendation', 'evidence') or not text or (typ == 'evidence' and not has_ev):
                continue
            try:
                v = ask(client, val_model, VAL.format(passage=r['passage'], evidence=r['evidence'], question=text), 200)
            except Exception as e:
                log('val fail', r['id'], e); continue
            if not all(v.get(x) is True for x in ('self_contained', 'answered_by_passage', 'not_copied')):
                dropped += 1; continue
            out.append(dict(id=f"{r['id']}-q{k}", doc=r['doc'], type=typ, question=text, answerable=True, source='synthetic',
                            gen_model=gen_model, val_model=val_model,
                            gold_spans=[[int(r['start']), int(r['end'])]],
                            evidence_span=[int(r['evidence_start']), int(r['evidence_end'])] if (has_ev and typ == 'evidence') else None))
    log(f'kept {len(out)}, dropped by validator {dropped}')
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='results/annotation/seed_passages.csv'); ap.add_argument('--out', default='results/questions_synth.jsonl')
    ap.add_argument('--gen-model', required=True); ap.add_argument('--val-model', required=True)
    ap.add_argument('--limit', type=int, default=0, help='first N passages only (pilot)')
    a = ap.parse_args()
    if a.gen_model == a.val_model:
        sys.exit('use DIFFERENT generator and validator models (otherwise the validator approves its own habits)')
    import anthropic
    rows = list(csv.DictReader(open(a.seeds, encoding='utf-8')))
    if a.limit: rows = rows[:a.limit]
    man = {r['slug']: r['lang'] for r in csv.DictReader(open('data/manifest.csv', encoding='utf-8'))}
    out = run(anthropic.Anthropic(), rows, a.gen_model, a.val_model, lambda d: 'русский' if man.get(d, 'ru') == 'ru' else 'English')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8') as f:
        for q in out: f.write(json.dumps(q, ensure_ascii=False) + '\n')
    print(a.out)
