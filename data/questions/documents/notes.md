# Question Generation — Issue Log and Analytical Summary

## 1. Frequent Technical Errors

| ID | Issue | Cause | Example | Fix |
|----|--------|--------|---------|-----|
| 1 | Missing `Year` in `columns_involved` | Forgot temporal dimension when counting | Comparing GDP growth over time but only counted “Country, GDP” | Auto-include `Year` when date field exists |
| 2 | Miscounted `rows_involved` | Counted matches instead of examined records | Filtered 5 records from 100 → recorded 5 instead of 100 | Always record examined rows |
| 3 | Arithmetic mistakes | Manual average ≈ wrong rounding | Mean(5, 7, 8) → wrong 7 instead of 6.67 | Python-based verification |
| 4 | Unanswerable over-explanation | Added paragraphs of reasoning in `answer` | “Cannot be determined because …” | Keep `answer` = "Cannot be determined" only |
| 5 | Proxy answers too verbose | Mixed reasoning and answer fields | “The trend shows X therefore …” | Move logic to `reasoning_steps` |

---

## 2. Structural and Design Problems
- **Forced table joins**: Trying to combine unrelated tables to reach quota.  
  → Now controlled by stop-signal step.  
- **Reasoning not human-like**: Steps written as code comments or mixed logic.  
  → Added template enforcing 3–5 natural-language steps.  
- **Template artifacts**: Copy-pasted placeholders left in `question` or `answer`.  
  → Added pre-submission linter.

---

## 3. Distribution and Diversity Imbalance
- Over-representation of *Proxy Inference* (~45%).  
- Under-representation of *Education* and *Demographics* (<10%).  
**Next iteration:** weighted sampling + cross-domain question encouragement.

---

## 4. Lessons Learned
1. Strict adherence to SOP dramatically improves QA pass rate.  
2. “Human-centric reasoning” remains subjective—peer review essential.  
3. Automatic validators prevent numeric errors but not logical incoherence.  
4. Tracker updates should be scheduled weekly, not continuous, to avoid merge conflicts.  
5. Maintain one central schema reference and one shared linter.
