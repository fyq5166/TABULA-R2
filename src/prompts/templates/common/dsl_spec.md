Refer to the DSL action space for available primitives and examples.
Use only:
- load(table_id) / load_as(table_id, alias) / save(alias) / use(alias)
- select(col1, col2, ...)
- filter(expr) / filter_all(expr1; expr2; ...) / filter_any(expr1; expr2; ...)
- derive(new_col = "A" + "B") with +,-,*,/
- groupby(...).agg({fn: col}) with sum, mean, max, min, count
- join(left, right, on=..., how=inner|left|right|outer)
- order(col1, ..., ascending=True)
- limit(n)

