DSL Action Space (Research-grade Spec)

**IMPORTANT: DSL operations work on REAL TABLES**
- All DSL commands operate on actual data tables loaded from the dataset
- The results of your DSL operations will be displayed in the user message history
- You can see the execution results and use them to make decisions
- Each operation modifies the active dataframe and the results are immediately available

This document defines the restricted DSL you must use inside the PLAN block. Each operator is explicit about semantics, constraints, and provides a minimal example. Column names must be taken exactly from the observation space. No network/filesystem/system calls are permitted.

Primitives

1) load(table_id)
- Effect: Load `data/tables/<table_id>/table.csv` into the active dataframe.
- Constraints: `table_id` must be a known id like `table_0093`.
- Example:
  PLAN
  load(table_0093)
  END PLAN

2) load_as(table_id, alias)
- Effect: Load a table and assign it to a named alias for later use or join.
- Example:
  PLAN
  load_as(table_0093, A)
  load_as(table_0106, B)
  END PLAN

3) save(alias)
- Effect: Save the current active dataframe into a named alias.
- Example:
  PLAN
  load(table_0093)
  save(A)
  END PLAN

4) use(alias)
- Effect: Switch the active dataframe to a previously saved alias.
- Example:
  PLAN
  use(A)
  END PLAN

5) select(col1, col2, ...)
- Effect: Keep only the listed columns, in order.
- Constraints: All columns must exist.
- Example:
  PLAN
  load(table_0093)
  select(Entity, Year, "Coal production (The SHIFT Project)")
  END PLAN

6) filter(expr)
- Effect: Row filter with a simple expression `<col> <op> <value>`.
- Ops: ==, !=, >, <, >=, <=
- Value: number or quoted string
- Example:
  PLAN
  load(table_0093)
  filter(Year == 2010)
  END PLAN

7) filter_all(expr1; expr2; ...)
- Effect: AND all expressions.
- Example:
  PLAN
  load(table_0093)
  filter_all(Year >= 2000; Entity == "United States")
  END PLAN

8) filter_any(expr1; expr2; ...)
- Effect: OR any expressions.
- Example:
  PLAN
  load(table_0093)
  filter_any(Entity == "United States"; Entity == "China")
  END PLAN

9) derive(new_col = "colA" + "colB") or derive(new_col = mean("colA"))
- Effect: Create a new column via a binary op between two columns or a column and a literal number, OR via aggregation functions.
- Ops: +, -, *, / for binary operations
- Aggregation functions: sum, mean, max, min, count, avg
- Example:
  PLAN
  load(table_0093)
  select(Year, "A", "B")
  derive(sumAB = "A" + "B")
  derive(avgA = mean("A"))
  END PLAN

**Common Pattern: Calculate sum of two columns**
- Example: Calculate total deaths from heart disease and cancer
  PLAN
  load(table_0001)
  filter_all(Entity == "United States"; Year == 1950)
  select("Heart disease - Deaths", "Cancers - Deaths")
  derive(total_deaths = "Heart disease - Deaths" + "Cancers - Deaths")
  select(total_deaths)
  END PLAN

10) groupby(col1, col2, ...).agg({fn: col, ...})
- Effect: Group by columns and aggregate with functions.
- Fns: sum, mean, max, min, count
- Example:
  PLAN
  load(table_0093)
  select(Year, "Coal production (The SHIFT Project)")
  groupby(Year).agg({sum: "Coal production (The SHIFT Project)"})
  END PLAN

11) join(left_alias, right_alias, on=col|col1,col2, how=inner|left|right|outer)
- Effect: Join two aliased dataframes on specified columns.
- Constraints: Both aliases must exist; join keys must exist in both.
- Example:
  PLAN
  load_as(table_0093, A)
  load_as(table_0106, B)
  join(A, B, on=Entity, how=inner)
  END PLAN

12) order(col1, col2, ..., ascending=True)
- Effect: Sort rows by columns.
- Example:
  PLAN
  load(table_0093)
  order(Year, ascending=False)
  END PLAN

13) limit(n)
- Effect: Return the first n rows.
- Example:
  PLAN
  load(table_0093)
  limit(10)
  END PLAN

General Constraints
- Use only the operators above; do not invent new ones.
- Column names are case-sensitive and must exactly match the observation space.
- If multiple tables are involved, join keys must be explicitly specified in the join command.
- Plans should be minimal and sufficient to compute the requested answer.

**CRITICAL DSL Syntax Rules**
- Column names with spaces MUST be in double quotes: "Heart disease - Deaths"
- Use derive() for calculations: derive(total = "A" + "B") NOT select("A" + "B")
- Filter expressions use semicolons: filter_all(Year >= 2000; Entity == "US")
- Join syntax: join(A, B, on=Entity, how=inner)
- Groupby syntax: groupby(Entity).agg({sum: "Column Name"})
- Order syntax: order("Column Name", ascending=False)

**Common DSL Errors to Avoid**
- WRONG: select("A" + "B") → CORRECT: derive(total = "A" + "B")
- WRONG: filter(Year >= 2000 AND Entity == "US") → CORRECT: filter_all(Year >= 2000; Entity == "US")
- WRONG: join(A, B, Entity) → CORRECT: join(A, B, on=Entity, how=inner)
- WRONG: groupby(Entity).sum("Column") → CORRECT: groupby(Entity).agg({sum: "Column"})
- WRONG: order(Column, desc) → CORRECT: order("Column", ascending=False)

**Real-World Error Examples and Fixes**
- WRONG: filter(Corruption Perception Index < 20) → CORRECT: filter("Corruption Perception Index" < 20)
- WRONG: select(CO2 emissions (gCO2/km)) → CORRECT: select("CO2 emissions (gCO2/km)")
- WRONG: derive(avg_gini = mean("Gini")) → CORRECT: groupby().agg({mean: "Gini"})
- WRONG: filter_all(Entity == "Canada"; Year >= 2010; Year <= 2014) → CORRECT: filter_all(Entity == "Canada"; "Year" >= 2010; "Year" <= 2014)
- WRONG: derive(average_estimate = mean("Chernobyl")) → CORRECT: groupby().agg({mean: "Chernobyl"})
- WRONG: order(average_estimate, ascending=False) END PLAN → CORRECT: order("average_estimate", ascending=False) END
- WRONG: END PLAN → CORRECT: END

**Column Name Handling**
- ALWAYS quote column names with spaces: "Column Name"
- The system will try to match column names even if quotes are missing
- Use exact column names from the table header
- For complex column names, copy them exactly from the table

**Media Analysis Example**
For media coverage analysis, you need to:
1. Load both tables: load(table_0001) and load_as(table_0065, media)
2. Calculate death share: derive(death_share = "Deaths" / sum("Deaths"))
3. Calculate media coverage: derive(media_share = "NYT" + "Guardian")
4. Find under-representation: derive(under_rep = death_share - media_share)
5. Order by under-representation: order("under_rep", ascending=False)

**Aggregation Example**
To calculate totals, use groupby().agg():
- WRONG: derive(total = sum("Column"))
- CORRECT: groupby().agg({sum: "Column"})
- For multiple columns: groupby().agg({sum: "Col1", mean: "Col2"})

