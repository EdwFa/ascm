import csv, importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location('gq', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'generate_questions.py'))
gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)


class Fake:
    class messages:
        @staticmethod
        def create(model, max_tokens, messages):
            class B: type = 'text'
            b = B()
            if 'Оцени' in messages[0]['content']:
                b.text = json.dumps({"self_contained": 'bad' not in messages[0]['content'], "answered_by_passage": True, "not_copied": True})
            else:
                b.text = json.dumps({"questions": [{"type": "recommendation", "question": "Q1?"}, {"type": "evidence", "question": "bad Q2?"}]})
            class R: content = [b]
            return R


def test_generate_and_filter():
    rows = [dict(id='x-0', doc='x', passage='p', evidence='e', start='1', end='5', evidence_start='6', evidence_end='9'),
            dict(id='x-1', doc='x', passage='p', evidence='', start='1', end='5', evidence_start='', evidence_end='')]
    out = gq.run(Fake(), rows, 'g', 'v', lambda d: 'English', log=lambda *a: None)
    assert [q['id'] for q in out] == ['x-0-q0', 'x-1-q0']          # 'bad' question dropped, evidence question not allowed without evidence
    assert out[0]['gold_spans'] == [[1, 5]] and out[0]['evidence_span'] is None and out[0]['source'] == 'synthetic'
