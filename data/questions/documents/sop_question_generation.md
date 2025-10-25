# Standard Operating Procedure (SOP) for Table-Based Question Generation

## 1. Purpose
This document provides a **step-by-step operational guide** for creating, validating, and maintaining table-based reasoning questions.  
It ensures that every generated question is:
- **Computationally verifiable**
- **Human-interpretable and diverse**
- **Fully documented for reproducibility**

---

## 2. Prerequisites

| Requirement | Description |
|--------------|-------------|
| **Data Access** | Tables stored under `data/tables/table_XXXX/` with corresponding `meta.json`. |
| **Environment** | Python ≥3.9 with pandas, numpy, and validator scripts installed. |
| **Reference Docs** | `metadata_definitions.md`, `question_template.json`, `table_distribution_tracker.md`. |
| **Trackers** | `group_tracker.md` (per thematic group) and `table_distribution_tracker.md` (global balance). |

---

## 3. Step-by-Step Procedure

### **Step 1 – Check for Stop Signals**
**Goal:** Ensure new questions add unique analytical value.  
**Input:** `group_tracker.md`  
**Actions:**  
1. Review prior questions in the same group.  
2. Stop generation if you detect:  
   - Repetitive reasoning structure  
   - Weak or forced table linkage  
   - Declining novelty or quality  
3. If triggered, record evidence and notify the QA lead.  
**Output:** Approval (✅ proceed / 🛑 stop).

---

### **Step 2 – Confirm Distribution Targets**
**Goal:** Keep dataset balanced.  
**Inputs:** `table_distribution_tracker.md`  
**Actions:**  
1. Check domain ratios → Health:Environment:Economics:Education:Demographics ≈ 2:2:1:0.5:0.5  
2. Check reasoning type ratios → Conditional:Arithmetic:Proxy ≈ 2:2:1  
3. Verify complexity balance across low (≤6.0), medium (6.1–21.5), high (>21.5).  
**Output:** Selected domain + reasoning type + complexity target.

---

### **Step 3 – Select Tables**
**Goal:** Choose 1 (table) or 2–3 (tables) supporting the reasoning goal.  
**Inputs:**  
- `data/table/table_index.json`  
- Each table’s `meta.json`  
**Actions:**  
1. Scan candidate tables by domain and complexity.  
2. For multi-table questions, ensure logical linkage (shared entity / time / metric).  
3. Avoid forced joins; prefer natural, meaningful relations.  
**Output:** List of selected tables + connection description.

---

### **Step 4 – Analyze Table Structure**
**Goal:** Understand the data before writing questions.  
**Actions:**  
1. Inspect columns (types, units, key identifiers).  
2. Check time range, granularity, and missingness.  
3. Identify potential reasoning opportunities (comparisons, correlations, changes).  
**Output:** Annotated notes on columns & patterns.

---

### **Step 5 – Design the Question**
**Goal:** Draft a complete, human-understandable question.  
**Actions:**  
1. Begin with “**Based on the table given, …**”.  
2. Ensure the structure fits one of the reasoning types:  
   - *Conditional Reasoning* → filtering & logic conditions  
   - *Arithmetic Aggregation* → sum / mean / ratio  
   - *Proxy Inference* → indirect evidence  
   - *Entity Alignment* → matching across tables  
3. Write 1–2 clear sentences; avoid ambiguous references.  
**Output:** Draft question text.

---

### **Step 6 – Compute the Answer**
**Goal:** Derive and verify the correct answer programmatically.  
**Actions:**  
1. Write a short Python snippet that reads the table(s).  
2. Perform necessary filtering/aggregation/inference.  
3. Record the final result (numeric, categorical, boolean, or list).  
4. If unanswerable, confirm by showing missing link or insufficient data.  
**Output:** Verified `answer`.

---

### **Step 7 – Write Reasoning Steps**
**Goal:** Provide human-readable explanation of how the answer is obtained.  
**Actions:**  
1. Outline **3–5 sequential steps** (concise, logical).  
2. Each step must correspond to an operation performed in Step 6.  
3. Avoid model-internal or algorithmic jargon.  
**Example:**  
1. Identify all years with available population data.  
2. Filter rows where GDP > 1000.  
3. Compare growth rates between 2010 and 2020.  
4. Conclude that Country X shows the largest increase.  
**Output:** Field `reasoning_steps`.

---

### **Step 8 – Calculate Complexity Metrics**
**Goal:** Quantify analytical difficulty.  
**Formula:**  
Complexity Score = (rows_involved × 0.1) + (columns_involved × 0.2) + (steps_count × 0.5)
**Actions:**  
1. Count `rows_involved` = records examined (not only matches).  
2. Count `columns_involved` = distinct columns required (always include `Year` if used).  
3. Record `steps_count` = number of reasoning steps.  
4. Compute and store `complexity_score`.  
**Output:** JSON field `complexity_metrics`.

---

### **Step 9 – Validate and Record**
**Goal:** Ensure correctness and compliance.  
**Actions:**  
1. Run automated validators for:  
   - JSON schema  
   - Answer reproducibility  
   - Complexity consistency  
2. Confirm naming (`question_XXX.json`) and placement under `data/questions/`.  
3. Review manually for readability and coherence.  
**Output:** Final validated question file.

---

### **Step 10 – Quality Assurance and Documentation**
**Goal:** Pass final gates and update trackers.  
**Actions:**  
1. Check all QA gates:  

| Gate | Criteria | Status |
|------|-----------|--------|
| Schema | All required fields present | ✅ |
| Answer Validation | Computationally verified | ✅ |
| Complexity | Formula correct | ✅ |
| Reasoning Steps | Human-like and complete | ✅ |
| Distribution | Within target quota | ✅ |

2. Update only **numerical counts** in:  
   - `group_tracker.md`  
   - `table_distribution_tracker.md`  
3. Log environment hashes and seeds in `run.meta.json`.  
4. Delete temporary validation scripts.  
**Output:** Published question + updated trackers.
