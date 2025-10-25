"""
Minimal DSL parser and executor for table reasoning.

Supported primitives (single-table MVP):
- load(table_id)
- load_as(table_id, alias)
- save(alias)
- use(alias)
- select(col1, col2, ...)
- filter(expr)  # expr: <col> <op> <value>; ops: ==, !=, >, <, >=, <=; value may be number or quoted string
- filter_all(expr1; expr2; ...)  # AND all
- filter_any(expr1; expr2; ...)  # OR any
- join(left_alias, right_alias, on=col|col1,col2, how=inner|left|right|outer)
- groupby(col1, col2, ...).agg({fn: col})  # fn in {sum, mean, max, min, count}
- order(col1, col2, ..., ascending=True)
- limit(n)
 - derive(new_col = "colA" + "colB")  # limited binary ops: +, -, *, /

Example plan:
    load(table_0093)
    select(Entity, Year, "Coal production (The SHIFT Project)")
    filter(Year == 2010)
    groupby().agg({sum: "Coal production (The SHIFT Project)"})
    limit(1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd


class ColumnNameError(Exception):
    """Special error for column name mismatches"""

    def __init__(
        self,
        requested_col: str,
        available_cols: List[str],
        suggestions: List[str] = None,
    ):
        self.requested_col = requested_col
        self.available_cols = available_cols
        # Convert to list to avoid pandas Index issues
        if suggestions is None:
            self.suggestions = []
        else:
            self.suggestions = (
                list(suggestions) if hasattr(suggestions, "__iter__") else []
            )
        super().__init__(f"Column '{requested_col}' not found")


DATA_ROOT = Path("data/tables")


@dataclass
class DSLCommand:
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _auto_quote_column_names(text: str) -> str:
    """Automatically add quotes to column names that contain spaces or special characters"""
    import re

    def quote_column_name(col_name):
        """为列名添加引号，如果包含空格或括号"""
        col_name = col_name.strip()
        if not col_name:
            return col_name

        # 如果已经有引号，直接返回
        if (col_name.startswith('"') and col_name.endswith('"')) or (
            col_name.startswith("'") and col_name.endswith("'")
        ):
            return col_name

        # 如果包含空格或括号，添加引号
        if " " in col_name or "(" in col_name:
            return f'"{col_name}"'

        return col_name

    # 逐行处理，避免复杂的正则表达式
    lines = text.split("\n")
    result_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            result_lines.append(line)
            continue

        # 处理order命令
        if line.startswith("order("):
            # 找到第一个逗号或右括号的位置
            paren_count = 0
            comma_pos = -1
            for i, char in enumerate(line):
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                elif char == "," and paren_count == 1:
                    comma_pos = i
                    break

            if comma_pos > 0:
                col_name = line[6:comma_pos].strip()  # 跳过'order('
                quoted_col = quote_column_name(col_name)
                result_lines.append(f"order({quoted_col}, ascending=False)")
            else:
                # 没有逗号，只有列名
                col_name = line[6:-1].strip()  # 跳过'order('和')'
                quoted_col = quote_column_name(col_name)
                result_lines.append(f"order({quoted_col}, ascending=False)")

        # 处理select命令
        elif line.startswith("select("):
            # 找到select命令的内容
            start = line.find("select(") + 7
            end = line.rfind(")")
            if start < end:
                content = line[start:end]
                # 简单的逗号分割
                parts = [part.strip() for part in content.split(",")]
                quoted_parts = [quote_column_name(part) for part in parts]
                result_lines.append(f'select({", ".join(quoted_parts)})')
            else:
                result_lines.append(line)

        # 处理filter命令
        elif line.startswith("filter("):
            # 使用正则表达式匹配filter命令
            match = re.match(r"filter\(([^=!<>]+?)\s*([=!<>]+)\s*([^)]+)\)", line)
            if match:
                col_name = match.group(1).strip()
                operator = match.group(2)
                value = match.group(3)
                quoted_col = quote_column_name(col_name)
                result_lines.append(f"filter({quoted_col} {operator} {value})")
            else:
                result_lines.append(line)

        # 处理filter_all命令
        elif line.startswith("filter_all("):
            # 找到filter_all命令的内容
            start = line.find("filter_all(") + 11
            end = line.rfind(")")
            if start < end:
                content = line[start:end]
                # 处理分号分隔的多个条件
                conditions = [cond.strip() for cond in content.split(";")]
                quoted_conditions = []
                for cond in conditions:
                    if cond:
                        # 简单的列名提取和加引号
                        parts = cond.split()
                        if len(parts) >= 3:
                            col_name = parts[0]
                            quoted_col = quote_column_name(col_name)
                            quoted_conditions.append(
                                cond.replace(col_name, quoted_col, 1)
                            )
                        else:
                            quoted_conditions.append(cond)
                result_lines.append(f'filter_all({"; ".join(quoted_conditions)})')
            else:
                result_lines.append(line)

        # 处理filter_any命令
        elif line.startswith("filter_any("):
            # 找到filter_any命令的内容
            start = line.find("filter_any(") + 11
            end = line.rfind(")")
            if start < end:
                content = line[start:end]
                # 处理分号分隔的多个条件
                conditions = [cond.strip() for cond in content.split(";")]
                quoted_conditions = []
                for cond in conditions:
                    if cond:
                        # 简单的列名提取和加引号
                        parts = cond.split()
                        if len(parts) >= 3:
                            col_name = parts[0]
                            quoted_col = quote_column_name(col_name)
                            quoted_conditions.append(
                                cond.replace(col_name, quoted_col, 1)
                            )
                        else:
                            quoted_conditions.append(cond)
                result_lines.append(f'filter_any({"; ".join(quoted_conditions)})')
            else:
                result_lines.append(line)

        # 处理groupby命令
        elif line.startswith("groupby("):
            # 找到groupby命令的内容
            start = line.find("groupby(") + 8
            end = line.rfind(")")
            if start < end:
                content = line[start:end]
                if content.strip():
                    # 简单的逗号分割
                    parts = [part.strip() for part in content.split(",")]
                    quoted_parts = [quote_column_name(part) for part in parts]
                    result_lines.append(f'groupby({", ".join(quoted_parts)})')
                else:
                    result_lines.append("groupby()")
            else:
                result_lines.append(line)

        # 处理derive命令
        elif line.startswith("derive("):
            # derive命令比较复杂，需要处理表达式中的列名
            # 这里只做简单的列名加引号，不处理复杂的表达式
            start = line.find("derive(") + 7
            end = line.rfind(")")
            if start < end:
                content = line[start:end]
                # 简单的处理：如果包含等号，处理等号后的部分
                if "=" in content:
                    parts = content.split("=", 1)
                    if len(parts) == 2:
                        left = parts[0].strip()
                        right = parts[1].strip()
                        # 对右边的表达式进行简单的列名加引号
                        # 这里只处理简单的列名引用，不处理复杂表达式
                        result_lines.append(line)  # 暂时不处理derive的复杂表达式
                    else:
                        result_lines.append(line)
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)

        # 处理join命令
        elif line.startswith("join("):
            # join命令通常不涉及列名加引号，直接返回
            result_lines.append(line)

        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def _find_fuzzy_column(target: str, available_cols: List[str]) -> Optional[str]:
    """Find the best matching column using conservative fuzzy matching with ambiguity handling"""
    target_clean = target.strip()

    # 1. Exact match (case insensitive) - highest priority
    for col in available_cols:
        if col.lower() == target_clean.lower():
            return col

    # 2. Remove quotes and try exact match
    target_no_quotes = target_clean.replace('"', "").replace("'", "").strip()
    for col in available_cols:
        if col.lower() == target_no_quotes.lower():
            return col

    # 3. Common patterns: handle "A" vs A, "Column Name" vs Column Name
    # Remove all quotes and normalize spaces
    target_normalized = " ".join(target_no_quotes.split())
    for col in available_cols:
        col_normalized = " ".join(col.split())
        if col_normalized.lower() == target_normalized.lower():
            return col

    # 4. Find all possible matches with scores
    target_words = set(target_normalized.lower().split())
    candidates = []

    for col in available_cols:
        col_words = set(col.lower().split())
        common_words = target_words.intersection(col_words)

        if len(common_words) > 0:
            # Calculate multiple scoring factors
            word_overlap = len(common_words) / len(
                target_words
            )  # How much of target is covered
            col_coverage = len(common_words) / len(
                col_words
            )  # How much of column is covered
            length_similarity = 1 - abs(len(target_normalized) - len(col)) / max(
                len(target_normalized), len(col)
            )

            # Combined score (weighted)
            combined_score = (
                word_overlap * 0.5 + col_coverage * 0.3 + length_similarity * 0.2
            )

            candidates.append((col, combined_score, len(common_words), word_overlap))

    if not candidates:
        return None

    # 5. Handle ambiguity: if multiple good matches, prefer the one with highest score
    # Sort by: 1) combined score, 2) number of common words, 3) word overlap
    candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)

    best_match, best_score, _, _ = candidates[0]

    # 6. Only return if score is above threshold
    if best_score > 0.3:  # At least 30% combined score
        return best_match

    return None


def _resolve_column_name(col_name: str, available_cols: List[str]) -> str:
    """Unified column name resolution with conservative fuzzy matching"""
    if col_name in available_cols:
        return col_name

    # Try conservative fuzzy matching
    fuzzy_match = _find_fuzzy_column(col_name, available_cols)
    if fuzzy_match:
        return fuzzy_match

    # Generate suggestions for better error messages
    suggestions = []
    col_lower = col_name.lower()
    for col in available_cols:
        if col_lower in col.lower() or col.lower() in col_lower:
            suggestions.append(col)

    # If no suggestions, show first few available columns
    if not suggestions:
        suggestions = available_cols[:5]

    raise ColumnNameError(col_name, available_cols, suggestions)


def parse_plan(plan_text: str, table_refs: List[str] = None) -> List[DSLCommand]:
    commands: List[DSLCommand] = []

    # Clean comments and natural language from plan
    plan_text = _clean_plan_text(plan_text)

    # Auto-quote column names to prevent common syntax errors
    plan_text = _auto_quote_column_names(plan_text)

    # Handle table name mapping for common placeholders
    plan_text = _map_table_names(plan_text, table_refs)

    # Handle descending parameter compatibility
    plan_text = _fix_descending_parameters(plan_text)

    lines = [
        ln.strip()
        for ln in plan_text.strip().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    for ln in lines:
        # groupby().agg({sum: col}) special case: may be on one line
        if ".agg(" in ln:
            gb_part, agg_part = ln.split(".agg(", 1)
            if not gb_part.startswith("groupby(") or not agg_part.endswith(")"):
                raise ValueError(f"Invalid groupby.agg syntax: {ln}")
            gb_args_raw = gb_part[len("groupby(") : -1].strip()
            gb_cols = (
                []
                if gb_args_raw == ""
                else [c.strip() for c in gb_args_raw.split(",") if c.strip()]
            )
            commands.append(DSLCommand("groupby", (gb_cols,), {}))

            # parse agg dict like {sum: "col"} or {sum: "c1", mean: "c2"}
            dict_raw = agg_part[:-1].strip()
            if not (dict_raw.startswith("{") and dict_raw.endswith("}")):
                raise ValueError(f"Invalid agg dict: {ln}")
            inner = dict_raw[1:-1].strip()
            # split by commas at top level
            parts = _split_args(inner)
            for part in parts:
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(f"Invalid agg entry: {part}")
                fn_str, col_str = part.split(":", 1)
                fn = fn_str.strip()
                col = _strip_quotes(col_str.strip())
                commands.append(DSLCommand("agg", (fn, col), {}))
            continue

        m = re.match(r"(\w+)\((.*)\)$", ln)
        if not m:
            raise ValueError(f"Invalid DSL line: {ln}")
        name = m.group(1)
        args_raw = m.group(2).strip()

        if name == "load":
            table_id = args_raw.strip()
            commands.append(DSLCommand("load", (table_id,), {}))
        elif name == "load_as":
            parts = _split_args(args_raw)
            if len(parts) != 2:
                raise ValueError(f"load_as expects (table_id, alias): {args_raw}")
            table_id = parts[0].strip()
            alias = parts[1].strip()
            commands.append(DSLCommand("load_as", (table_id, alias), {}))
        elif name == "save":
            alias = args_raw.strip()
            commands.append(DSLCommand("save", (alias,), {}))
        elif name == "use":
            alias = args_raw.strip()
            commands.append(DSLCommand("use", (alias,), {}))
        elif name == "select":
            cols = [c.strip() for c in _split_args(args_raw)] if args_raw else []
            cols = [_strip_quotes(c) for c in cols]
            commands.append(DSLCommand("select", (cols,), {}))
        elif name == "filter":
            # very simple expr: <col> <op> <value>
            expr = args_raw
            commands.append(DSLCommand("filter", (expr,), {}))
        elif name == "filter_all":
            # exprs separated by ';' ANDed
            exprs = [e.strip() for e in args_raw.split(";") if e.strip()]
            commands.append(DSLCommand("filter_all", (exprs,), {}))
        elif name == "filter_any":
            # exprs separated by ';' ORed
            exprs = [e.strip() for e in args_raw.split(";") if e.strip()]
            commands.append(DSLCommand("filter_any", (exprs,), {}))
        elif name == "derive":
            # derive(new = "colA" + "colB")
            if "=" not in args_raw:
                raise ValueError(f"Invalid derive, expected name = expr: {args_raw}")
            left, right = args_raw.split("=", 1)
            new_col = left.strip()
            expr = right.strip()

            # Aggregation functions are now supported in derive

            commands.append(DSLCommand("derive", (new_col, expr), {}))
        elif name == "groupby":
            gb_cols = (
                [] if args_raw == "" else [c.strip() for c in _split_args(args_raw)]
            )
            gb_cols = [_strip_quotes(c) for c in gb_cols]
            commands.append(DSLCommand("groupby", (gb_cols,), {}))
        elif name == "join":
            # join(left_alias, right_alias, on=col|col1,col2, how=inner)
            parts = _split_args(args_raw)
            if len(parts) < 2:
                raise ValueError(
                    f"join expects at least (left_alias, right_alias): {args_raw}"
                )
            left_alias = parts[0].strip()
            right_alias = parts[1].strip()
            on_cols: List[str] = []
            how = "inner"
            if len(parts) > 2:
                tail = ",".join(parts[2:])
                m_on = re.search(r"on\s*=\s*([^,]+)", tail, re.IGNORECASE)
                if m_on:
                    on_raw = m_on.group(1).strip()
                    on_cols = [c.strip() for c in on_raw.split(",") if c.strip()]
                    on_cols = [_strip_quotes(c) for c in on_cols]
                m_how = re.search(
                    r"how\s*=\s*(inner|left|right|outer)", tail, re.IGNORECASE
                )
                if m_how:
                    how = m_how.group(1).lower()
            commands.append(
                DSLCommand(
                    "join", (left_alias, right_alias), {"on": on_cols, "how": how}
                )
            )
        elif name == "order":
            # order(col1, col2, ..., ascending=True)
            ascending = True
            asc_match = re.search(
                r"ascending\s*=\s*(True|False)", args_raw, re.IGNORECASE
            )
            if asc_match:
                ascending = asc_match.group(1).lower() == "true"
                args_core = re.sub(
                    r",?\s*ascending\s*=\s*(True|False)",
                    "",
                    args_raw,
                    flags=re.IGNORECASE,
                ).strip()
            else:
                args_core = args_raw
            cols = [
                _strip_quotes(c.strip()) for c in _split_args(args_core) if c.strip()
            ]
            commands.append(DSLCommand("order", (cols,), {"ascending": ascending}))
        elif name == "limit":
            n = int(args_raw)
            commands.append(DSLCommand("limit", (n,), {}))
        else:
            raise ValueError(f"Unknown DSL command: {name}")
    return commands


def _split_args(arg_str: str) -> List[str]:
    # split by commas but respect quotes
    parts: List[str] = []
    cur = []
    in_quote: Optional[str] = None
    for ch in arg_str:
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            cur.append(ch)
        elif ch == "," and in_quote is None:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _eval_filter_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    # Support simple: <col> <op> <value>
    m = re.match(r"\s*(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)\s*", expr)
    if not m:
        raise ValueError(f"Unsupported filter expr: {expr}")
    col, op, val_raw = m.group(1).strip(), m.group(2), m.group(3).strip()
    col = _strip_quotes(col)
    if val_raw.startswith('"') or val_raw.startswith("'"):
        val = _strip_quotes(val_raw)
    else:
        try:
            val = float(val_raw) if "." in val_raw else int(val_raw)
        except ValueError:
            # treat as string
            val = val_raw
    try:
        col = _resolve_column_name(col, df.columns)
    except ColumnNameError as e:
        raise KeyError(
            f"Column not found: {col}. Available columns: {e.available_cols[:5]}"
        )
    if op == "==":
        return df[col] == val
    if op == "!=":
        return df[col] != val
    if op == ">":
        return df[col] > val
    if op == "<":
        return df[col] < val
    if op == ">=":
        return df[col] >= val
    if op == "<=":
        return df[col] <= val
    raise ValueError(f"Unsupported operator: {op}")


def _split_binary_expr(expr: str) -> Tuple[str, str, str]:
    # Split on first top-level operator (+ - * /) not inside quotes
    in_quote: Optional[str] = None
    for idx, ch in enumerate(expr):
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch in ["+", "-", "*", "/"] and in_quote is None:
            left = expr[:idx].strip()
            op = ch
            right = expr[idx + 1 :].strip()
            return left, op, right
    raise ValueError(f"Unsupported derive expr (no top-level operator): {expr}")


def execute_plan(plan_text: str, table_refs: List[str] = None) -> pd.DataFrame:
    try:
        cmds = parse_plan(plan_text, table_refs)
        df: pd.DataFrame | None = None
        gb_cols: Optional[List[str]] = None
        named: Dict[str, pd.DataFrame] = {}

        i = 0
        while i < len(cmds):
            cmd = cmds[i]
            if cmd.name == "load":
                table_id = cmd.args[0]
                csv_path = DATA_ROOT / table_id / "table.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(f"Table not found: {csv_path}")
                df = pd.read_csv(csv_path)
                i += 1
                continue
            elif cmd.name == "load_as":
                table_id, alias = cmd.args
                csv_path = DATA_ROOT / table_id / "table.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(f"Table not found: {csv_path}")
                df = pd.read_csv(csv_path)
                named[alias] = df.copy()
                i += 1
                continue
            elif cmd.name == "save":
                alias = cmd.args[0]
                assert df is not None, "save before any dataframe"
                named[alias] = df.copy()
                i += 1
                continue
            elif cmd.name == "use":
                alias = cmd.args[0]
                if alias not in named:
                    raise KeyError(f"Unknown alias: {alias}")
                df = named[alias].copy()
                i += 1
                continue
            elif cmd.name == "select":
                assert df is not None, "select before load"
                cols: List[str] = cmd.args[0]
                # Use unified column resolution
                resolved_cols = []
                for c in cols:
                    try:
                        resolved_col = _resolve_column_name(c, df.columns)
                        resolved_cols.append(resolved_col)
                    except ColumnNameError as e:
                        available_list = list(e.available_cols)[:5]
                        raise KeyError(
                            f"Column not found: {c}. Available columns: {available_list}"
                        )
                df = df[resolved_cols]
                i += 1
                continue
            elif cmd.name == "filter":
                assert df is not None, "filter before load"
                mask = _eval_filter_expr(df, cmd.args[0])
                df = df[mask]
                i += 1
                continue
            elif cmd.name == "filter_all":
                assert df is not None, "filter_all before load"
                exprs = cmd.args[0]
                mask = None
                for e in exprs:
                    m = _eval_filter_expr(df, e)
                    if mask is None:
                        mask = m
                    else:
                        mask = mask & m
                df = df[mask] if mask is not None else df
                i += 1
                continue
            elif cmd.name == "filter_any":
                assert df is not None, "filter_any before load"
                exprs = cmd.args[0]
                mask = None
                for e in exprs:
                    m = _eval_filter_expr(df, e)
                    if mask is None:
                        mask = m
                    else:
                        mask = mask | m
                df = df[mask] if mask is not None else df
                i += 1
                continue
            elif cmd.name == "derive":
                assert df is not None, "derive before load"
                new_col, expr = cmd.args

                # Check if this is an aggregation function
                agg_functions = ["sum", "mean", "max", "min", "count", "avg"]
                is_aggregation = False
                agg_func = None
                agg_col = None

                for func in agg_functions:
                    if f"{func}(" in expr.lower():
                        is_aggregation = True
                        agg_func = func
                        # Extract column name from function call
                        import re

                        # Support both quoted and unquoted column names
                        match = re.search(
                            rf'{func}\s*\(\s*"([^"]+)"\s*\)', expr, re.IGNORECASE
                        )
                        if not match:
                            # Try without quotes
                            match = re.search(
                                rf"{func}\s*\(\s*([^)]+)\s*\)", expr, re.IGNORECASE
                            )
                        if match:
                            agg_col = match.group(1).strip('"')
                        break

                if is_aggregation and agg_col:
                    # Handle aggregation function
                    try:
                        agg_col = _resolve_column_name(agg_col, df.columns)
                        if agg_func == "sum":
                            df[new_col] = df[agg_col].sum()
                        elif agg_func == "mean":
                            df[new_col] = df[agg_col].mean()
                        elif agg_func == "max":
                            df[new_col] = df[agg_col].max()
                        elif agg_func == "min":
                            df[new_col] = df[agg_col].min()
                        elif agg_func == "count":
                            df[new_col] = df[agg_col].count()
                        elif agg_func == "avg":
                            df[new_col] = df[agg_col].mean()
                        else:
                            raise ValueError(
                                f"Unsupported aggregation function: {agg_func}"
                            )
                    except ColumnNameError as e:
                        raise KeyError(
                            f"Column not found: {agg_col}. Available columns: {e.available_cols[:5]}"
                        )
                    except Exception as e:
                        raise ValueError(f"Error in derive aggregation: {e}")
                else:
                    # Handle binary operations (original logic)
                    a_raw, op, b_raw = _split_binary_expr(expr)

                    def parse_term(t: str):
                        if (t.startswith('"') and t.endswith('"')) or (
                            t.startswith("'") and t.endswith("'")
                        ):
                            col = _strip_quotes(t)
                            try:
                                col = _resolve_column_name(col, df.columns)
                            except ColumnNameError as e:
                                raise KeyError(
                                    f"Column not found: {col}. Available columns: {e.available_cols[:5]}"
                                )
                            return df[col]
                        try:
                            return float(t) if "." in t else int(t)
                        except ValueError:
                            col = _strip_quotes(t)
                            try:
                                col = _resolve_column_name(col, df.columns)
                                return df[col]
                            except ColumnNameError as e:
                                raise ValueError(
                                    f"Invalid derive term: {t}. Available columns: {e.available_cols[:5]}"
                                )

                    a = parse_term(a_raw)
                    b = parse_term(b_raw)
                    if op == "+":
                        df[new_col] = a + b
                    elif op == "-":
                        df[new_col] = a - b
                    elif op == "*":
                        df[new_col] = a * b
                    elif op == "/":
                        df[new_col] = a / b
                    else:
                        raise ValueError(f"Unsupported derive operator: {op}")
                i += 1
                continue
            elif cmd.name == "groupby":
                assert df is not None, "groupby before load"
                gb_cols = cmd.args[0]
                # groupby with empty list means no group keys (aggregate over all rows)
                i += 1
                continue
            elif cmd.name == "join":
                left_alias, right_alias = cmd.args
                on_cols = cmd.kwargs.get("on", [])
                how = cmd.kwargs.get("how", "inner")
                if left_alias not in named or right_alias not in named:
                    raise KeyError(
                        f"Unknown alias in join: {left_alias}, {right_alias}"
                    )
                left_df = named[left_alias]
                right_df = named[right_alias]
                if not on_cols:
                    raise ValueError("join requires on=... columns")
                for c in on_cols:
                    if c not in left_df.columns or c not in right_df.columns:
                        raise KeyError(f"Join key not found: {c}")
                df = left_df.merge(right_df, on=on_cols, how=how)
                i += 1
                continue
            elif cmd.name == "agg":
                assert df is not None, "agg before load"
                assert gb_cols is not None, "agg requires preceding groupby"
                # collect contiguous agg cmds
                agg_items = []
                j = i
                while j < len(cmds) and cmds[j].name == "agg":
                    fn, col = cmds[j].args
                    fn = fn.lower()
                    try:
                        col = _resolve_column_name(col, df.columns)
                    except ColumnNameError as e:
                        raise KeyError(
                            f"Column not found: {col}. Available columns: {e.available_cols[:5]}"
                        )
                    if fn not in {"sum", "mean", "max", "min", "count"}:
                        raise ValueError(f"Unsupported agg fn: {fn}")
                    agg_items.append((fn, col))
                    j += 1

                # perform aggregation
                if len(gb_cols) == 0:
                    out_dict = {}
                    for fn, col in agg_items:
                        out_dict[f"{fn}_{col}"] = [getattr(df[col], fn)()]
                    df = pd.DataFrame(out_dict)
                else:
                    grouped = df.groupby(gb_cols, dropna=False)
                    agg_spec: Dict[str, List[str]] = {}
                    for fn, col in agg_items:
                        agg_spec.setdefault(col, []).append(fn)
                    out = grouped.agg(agg_spec)
                    # Flatten MultiIndex columns
                    out.columns = [f"{fn}_{col}" for col, fn in out.columns]
                    df = out.reset_index()
                gb_cols = None
                i = j
                continue
            elif cmd.name == "order":
                assert df is not None, "order before load"
                cols = cmd.args[0]
                ascending = cmd.kwargs.get("ascending", True)
                for c in cols:
                    if c not in df.columns:
                        raise KeyError(f"Column not found: {c}")
                df = df.sort_values(by=cols, ascending=ascending)
                i += 1
                continue
            elif cmd.name == "limit":
                assert df is not None, "limit before load"
                n = int(cmd.args[0])
                df = df.head(n)
                i += 1
                continue
            else:
                raise ValueError(f"Unsupported command at execution: {cmd.name}")

        assert df is not None, "Empty plan produced no DataFrame"
        return df.reset_index(drop=True)
    except ColumnNameError as e:
        # Convert ColumnNameError to a special error that can be caught by the caller
        error_msg = f"COLUMN_NAME_ERROR: {e.requested_col} not found. Available: {e.available_cols[:5]}"
        raise ValueError(error_msg)
    except Exception as e:
        # Add more detailed error information
        error_msg = f"EXEC_ERROR: {str(e)}"
        raise ValueError(error_msg)


def _clean_plan_text(plan_text: str) -> str:
    """Clean comments and natural language from plan text"""
    lines = plan_text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove line comments
        if "#" in line:
            line = line.split("#")[0].strip()
            if not line:
                continue

        # Check for natural language patterns
        natural_language_patterns = [
            r"^please\s+",
            r"^let\s+me\s+",
            r"^i\s+will\s+",
            r"^this\s+plan\s+",
            r"^assuming\s+",
            r"^i\s+understand\s+",
            r"^thank\s+you",
            r"^correct\s+or\s+not",
            r"^if\s+this\s+attempt",
            r"^you\s+must\s+try",
            r"^consider\s*:",
            r"^do\s+not\s+repeat",
            r"^simpler\s+operations",
            r"^different\s+column",
            r"^basic\s+select",
            r"^check\s+table",
        ]

        is_natural_language = any(
            re.search(pattern, line, re.IGNORECASE)
            for pattern in natural_language_patterns
        )

        if is_natural_language:
            # This is natural language, skip it
            continue

        # Check if line contains only DSL commands
        if re.match(
            r"^(load|select|filter|groupby|order|limit|derive|save|use|join)\s*\(", line
        ):
            cleaned_lines.append(line)
        elif line.startswith("groupby(") and ".agg(" in line:
            cleaned_lines.append(line)
        elif line.startswith("filter_all(") or line.startswith("filter_any("):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _map_table_names(plan_text: str, table_refs: List[str] = None) -> str:
    """Map common table placeholders to actual table names"""
    if not table_refs:
        return plan_text

    # Common placeholder patterns
    placeholder_patterns = [
        r"\btable_T\b",
        r"\btable\b(?!_\d+)",  # table but not table_123
        r"\btable_\?\b",
    ]

    result = plan_text

    # If only one table, map all placeholders to it
    if len(table_refs) == 1:
        target_table = table_refs[0]
        for pattern in placeholder_patterns:
            result = re.sub(pattern, target_table, result)
    # If multiple tables, map to first one but warn
    elif len(table_refs) > 1:
        target_table = table_refs[0]
        for pattern in placeholder_patterns:
            result = re.sub(pattern, target_table, result)

    return result


def _fix_descending_parameters(plan_text: str) -> str:
    """Convert descending=True to ascending=False for compatibility"""
    # Pattern: order(col, descending=True) -> order(col, ascending=False)
    result = re.sub(
        r"order\(([^)]+),\s*descending\s*=\s*True\)",
        r"order(\1, ascending=False)",
        plan_text,
        flags=re.IGNORECASE,
    )

    # Pattern: order(col, descending=False) -> order(col, ascending=True)
    result = re.sub(
        r"order\(([^)]+),\s*descending\s*=\s*False\)",
        r"order(\1, ascending=True)",
        result,
        flags=re.IGNORECASE,
    )

    return result


if __name__ == "__main__":
    # Simple smoke test: sum coal production in 2010 from table_0093
    plan = (
        "load(table_0093)\n"
        'select(Year, "Coal production (The SHIFT Project)")\n'
        "filter(Year == 2010)\n"
        'groupby().agg({sum: "Coal production (The SHIFT Project)"})\n'
        "limit(1)\n"
    )
    out = execute_plan(plan)
    print(out)
