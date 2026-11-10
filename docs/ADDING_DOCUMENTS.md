# Как добавить документы в набор

1. **PDF** положите в `data/pdf/`, **готовый текст** (русские `.md`) в `data/raw/`. Содержимое этих папок не коммитится, если не нужно.
2. Добавьте строку в `data/manifest.csv`:

| Колонка | Значения |
|---|---|
| `slug` | короткое латинское имя, без пробелов |
| `path` | `data/raw/<slug>.md` или `data/raw_en/<slug>.txt` |
| `pdf` | имя PDF в `data/pdf/` (только для английских документов) |
| `lang` | `ru` или `en` |
| `family` | формат оценки доказательности: `ru_uur`, `esmo`, `asco`, `eras` или `none` (если оценок нет или формат ещё не поддержан) |
| `kind` | `guideline`, `guideline_summary`, `position_paper`, `review` |
| `publisher`, `licence_note` | издатель и условия использования, проверьте лицензию |
| `role` | `dev`, `test` или `excluded` |

3. Для PDF выполните `python scripts/extract_pdfs.py`: он удаляет водяные знаки (в PDF ASCO встроен IP-адрес скачавшего), колонтитулы и список литературы.
4. `python scripts/build_corpus.py --tokenizer bge-m3`, затем `python scripts/run_experiment.py ...`.
5. Проверьте в `results/corpus_stats.csv` колонки `evidence_sentences` и `pairs`: если у руководства с градациями там 0, детектор формата оценки не сработал, и нужен новый семейство в `src/ascm/loc.py` (`FAMILIES`).

Не добавляйте обзорные статьи и документы без рекомендаций в основной набор: для них нельзя оценить ERL.
