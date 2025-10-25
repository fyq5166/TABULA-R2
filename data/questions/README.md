# data/questions Directory

## Overview
This directory contains the hand-curated question repository for TABULA-R2 Bench. It spans three complementary subsets—single-table, multi-table, and distractor-augmented—designed to evaluate table reasoning capabilities across domains, reasoning types, and difficulty tiers. Each question is serialized as a JSON file with standardized metadata to support downstream evaluation and stratified analyses.

## Directory Layout
- `single_table/` — Finalized questions referencing exactly one table. Coverage tracking lives in `single_table/table_distribution_tracker.md`.
- `multi_table/` — Questions requiring joins or cross-table alignment within a thematic cohort. Each cohort subdirectory (e.g., `Nutrition_and_Dietary_Health/`) must contain a `group_tracker.md` derived from `multi_table/group_tracker_template.md`. Global coverage is summarized in `multi_table/distribution_tracker.md`, with automation in `multi_table/update_distribution_tracker.py`.
- `distractor_bank/` — Robustness variants of selected base questions augmented with irrelevant or misleading tables. Each subfolder mirrors the base `question_XXX` identifier and stores JSON variants annotated with distractor metadata.
- `documents/` — Authoring SOPs, schema references, and templates:
  - [`documents/sop_question_generation.md`](documents/sop_question_generation.md)
  - [`documents/question_template.json`](documents/question_template.json)
  - [`documents/metadata_definitions.md`](documents/metadata_definitions.md)
  - [`documents/notes.md`](documents/notes.md)

## Naming and Normalization
- File naming: `question_XXX.json` (zero-padded; e.g., `question_001.json`).
- Internal identifier: `question_id` must match the normalized form `question_XXX`.
- Cohort folders: Title-cased words joined with underscores (e.g., `Energy_Economics_and_Resources/`).
- Note: In `distractor_bank/`, ensure tracker file naming is `distribution_tracker.md` (correct spelling) to avoid duplicates or drift.

## Question Schema
Required fields (shared across single-/multi-table and distractor variants):
- `question_id` (string): Normalized identifier `question_XXX`.
- `question` (string): Natural language prompt.
- `table_refs` (string[]): Table identifiers referenced by the question. Single-table ⇒ length 1; multi-table ⇒ length ≥ 2.
- `answer` (string | number | boolean | object): Ground-truth answer matching `answer_type`.
- `answer_type` (string enum): `numerical` | `boolean` | `categorical` | `text` | `NA`.
- `domain` (string): Canonical domain; must be consistent with `data/tables/table_index.json`.
- `reasoning_type` (string enum): `arithmetic_aggregation` | `conditional_reasoning` | `entity_alignment` | `proxy_inference`.
- `answerable` (bool): Whether the question is answerable from the referenced tables.
- `complexity_metrics` (object): { `rows_involved`: int, `columns_involved`: int, `steps_count`: int, `complexity_score`: float }.

Optional fields:
- `notes` (string): Author remarks for curation.
- Distractor variants:
  - `has_distractor` (bool = true)
  - `distractor_type` (enum): `irrelevant` | `misleading` | `relevant`

Minimal examples:
```json
// single_table
{
  "question_id": "question_001",
  "question": "What is the total number of deaths in 1950?",
  "table_refs": ["table_0001"],
  "answer": 746438,
  "answer_type": "numerical",
  "domain": "health",
  "reasoning_type": "arithmetic_aggregation",
  "answerable": true,
  "complexity_metrics": { "rows_involved": 1, "columns_involved": 2, "steps_count": 1, "complexity_score": 2.9 }
}

// multi_table
{
  "question_id": "question_236",
  "question": "Did country A's metric exceed country B's in 2010?",
  "table_refs": ["table_0071", "table_0093"],
  "answer": true,
  "answer_type": "boolean",
  "domain": "environment",
  "reasoning_type": "entity_alignment",
  "answerable": true,
  "complexity_metrics": { "rows_involved": 2, "columns_involved": 3, "steps_count": 3, "complexity_score": 3.1 }
}

// distractor variant
{
  "question_id": "question_236",
  "question": "Did country A's metric exceed country B's in 2010?",
  "table_refs": ["table_0071", "table_0093", "table_0005"],
  "answer": true,
  "answer_type": "boolean",
  "domain": "environment",
  "reasoning_type": "entity_alignment",
  "answerable": true,
  "has_distractor": true,
  "distractor_type": "irrelevant",
  "complexity_metrics": { "rows_involved": 2, "columns_involved": 3, "steps_count": 3, "complexity_score": 3.1 }
}
```

## Complexity Scoring
We adopt a simple, reproducible scoring model aligned with the experiments plan:

\[ complexity\_score = (rows\_involved \times 0.1) + (columns\_involved \times 0.2) + (steps\_count \times 0.5) \]

Complexity tiers:
- Low (≤ 6.0), Medium (6.1–21.5), High (> 21.5)

The `complexity_metrics` object in each question file must reflect the above definition to enable stratified sampling and difficulty analyses.

## Workflow & Quality Controls
1. Cohort stop-signal review: Confirm novelty; avoid forced joins and redundant patterns (`multi_table/*/group_tracker.md`).
2. Distribution planning: Check `multi_table/distribution_tracker.md` and `single_table/table_distribution_tracker.md` to target domain, reasoning type, and tier.
3. Table inspection: Use `data/tables/table_index.json` and per-table `meta.json` to understand schema, time ranges, and alignment.
4. Authoring: Follow [`documents/sop_question_generation.md`](documents/sop_question_generation.md) and use [`documents/question_template.json`](documents/question_template.json) in accordance with [`documents/metadata_definitions.md`](documents/metadata_definitions.md).
5. Programmatic verification: Compute gold answers, validate `answer_type`, and confirm `complexity_metrics`.
6. Human QA and publication: Finalize JSON, update trackers, and place files under the appropriate subset.
7. Retrospective logging: Record recurring issues (temporal mismatches, miscounts, verbosity) to refine SOPs and validators.

Common failure modes:
- `question_id` and filename mismatch; domain mismatch with table index; inconsistent `answer_type` vs `answer`; missing or out-of-range `complexity_metrics`.

## Coupling with Runner & Config
- Selection configuration (`configs/qna.yaml`):
  - `selection.mode`: `all | single_table | multi_table | distractor | custom`
  - `selection.custom_ids`: supports explicit IDs and ranges, e.g., `["1-3", "4"]`
- Observation space (`qna.obs`): all columns visible; row previews depend on `header_5 | header_1 | full_table` with a cap.
- Validator outputs are persisted alongside runs; question files must contain accurate gold answers for LLM validation.

## Sanity Check & Analysis
Run a quick distribution analysis after large edits:
```bash
uv run python experiments/analysis/question_statistics.py --output_dir experiments/analysis/question_results --generate_plots
```

## Related Files
- `data/tables/table_index.json` — canonical table metadata and domain mapping.
- `configs/qna.yaml` — selection, observation space, prompt, run, and validator parameters.
- `experiments/PLAN.md` — experimental design and stratification dimensions.
