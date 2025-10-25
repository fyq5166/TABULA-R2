#!/usr/bin/env python3
"""
Q&A Runner (config-driven scaffold)

Purpose:
- Entry point for running Q&A agent evaluations over curated questions.
- Loads parameters from configs/qna.yaml via src.utils.config.load_config.
- To be extended with prompt wiring, question loading, LLM invocation, and result aggregation/visualization.
"""

import argparse
import sys
from pathlib import Path
import time
import json
import re
import subprocess

# Allow running as a script by adding project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_config_value
from src.evaluation.question_selector import SelectionConfig, collect_question_records
from src.evaluation.observation_space import ObservationSpaceGenerator
from src.prompts.prompt_manager import prompt_manager
from src.utils.llm_client import LLMClient

# Enable optional DSL execution in multi-round flow
from src.evaluation.dsl_executor import execute_plan, parse_plan
from src.evaluation.log_writer import write_jsonl_append, write_json_pretty
from src.evaluation.plan_parser import parse_plan_or_end
from src.evaluation.guidance import guidance_for
from src.evaluation.llm_validator import LLMValidator
from src.utils.logging_utils import setup_logging, create_session_log_file, get_logger


def _extract_plan_block(text: str) -> str | None:
    """Extract the DSL/Pandas PLAN block between 'PLAN' and 'END PLAN'."""
    if not text:
        return None
    # Try fenced variants first
    m = re.search(r"PLAN\s*\n([\s\S]*?)\nEND PLAN", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: whole text if it looks like commands
    if any(k in text for k in ("load_as(", "select(", "filter(", "groupby(", "join(")):
        return text.strip()
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q&A agent over curated questions")
    parser.add_argument(
        "--config", type=str, default="configs/qna.yaml", help="Path to Q&A YAML config"
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        help="Enable continue mode (overrides config)",
    )
    parser.add_argument(
        "--continue-from", type=int, help="Question number to continue from (1-based)"
    )
    parser.add_argument(
        "--continue-batches",
        action="store_true",
        help="Continue to next batch after current batch completes (overrides config)",
    )
    return parser.parse_args(argv)


def _read_action_space() -> str:
    p = (
        PROJECT_ROOT
        / "src"
        / "prompts"
        / "templates"
        / "qna_execution"
        / "dsl_action_space.md"
    )
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return "(action space unavailable)"


def _ensure_model_pulled(model_name: str, llm_url: str) -> None:
    """Ensure the Ollama model is available locally by pulling if missing.
    Only attempts pull when targeting localhost."""
    try:
        import requests  # local import to avoid global dependency issues

        resp = requests.get(f"{llm_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if any(model_name in m for m in models):
            return
    except Exception:
        # If service not reachable, still try a pull when localhost
        pass
    if "localhost" in llm_url or "127.0.0.1" in llm_url:
        print(f"[QnA] Pulling model {model_name} via 'ollama pull' ...")
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"[QnA] Model {model_name} pulled successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[QnA] Failed to pull model {model_name}: {e}")
    else:
        print(f"[QnA] Skipping local pull for non-local llm_url={llm_url}")


def _build_force_plan_prompt(question_id: str, table_refs: list[str]) -> str:
    dsl_spec = _read_action_space()
    tables = ", ".join(table_refs or [])
    return (
        f"System: You must output exactly one PLAN block using the DSL below. No other text.\n\n"
        f"DSL Action Space:\n{dsl_spec}\n\n"
        f"User: question_id={question_id} table_refs=[{tables}]\n"
        f"Instruction: Produce PLAN only.\n"
        f"Format:\nPLAN\n<dsl commands>\nEND PLAN\n"
    )


def _build_force_end_prompt(question_id: str, short_answer_hint: str) -> str:
    return (
        "System: You must output exactly one END block with the final short answer. No other text.\n\n"
        f"User: question_id={question_id}\n"
        f"Hint: If you know the concise final answer, output it now.\n"
        f"Format:\nEND\n<final short answer only>\nEND\n"
        f"Example: END\\n42\\nEND\n"
    )


_guidance_for = guidance_for


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    # Load Q&A configuration (disable schema validation – qna.yaml is custom)
    config = load_config(args.config, validate_schema=False)

    # Initialize logging under configured logs folder using utils naming
    logs_root = Path(get_config_value(config, "qna.result.log_dir", "logs"))
    # Always use datestamp-based plain filename (YYYYMMDD_HHMMSS.log)
    try:
        session_log = create_session_log_file(str(logs_root), plain_name=True)
    except TypeError:
        session_log = create_session_log_file(str(logs_root))
    # Mirror logs to both file and terminal so users can see live progress
    setup_logging(
        log_level="DEBUG",
        log_file=str(session_log),
        console_output=True,
        verbose=False,
        plain_file=True,
        console_level="INFO",
        file_level="DEBUG",
    )
    logger = get_logger("qna_runner")
    logger.info("======================================================")
    logger.info("      UNIFIED Q-A RUNNER & EVALUATOR STARTING  ")
    logger.info("======================================================")

    questions_path = Path(
        get_config_value(config, "qna.questions_path", "data/questions")
    )
    model_name = get_config_value(config, "qna.model", "llama3")
    limit = get_config_value(config, "qna.limit", None)
    result_root = Path(
        get_config_value(config, "qna.result.result_dir", "experiments/results")
    )
    seed = int(get_config_value(config, "qna.seed", 42))
    llm_url = get_config_value(config, "qna.llm.url", "http://localhost:11434")
    sel_mode = get_config_value(config, "qna.selection.mode", "all")
    custom_ids = list(get_config_value(config, "qna.selection.custom_ids", []))
    batch_size = int(get_config_value(config, "qna.selection.batch_size", 10))
    continue_mode = bool(get_config_value(config, "qna.selection.continue", False))
    continue_from = get_config_value(config, "qna.selection.continue_from", None)

    # Override with command line arguments if provided
    if getattr(args, "continue", False):
        continue_mode = True
    if getattr(args, "continue_from", None) is not None:
        continue_from = getattr(args, "continue_from")

    # Initialize validator
    validator_model = get_config_value(
        config, "qna.validator.model", "mock-always-true"
    )
    validator_timeout = float(get_config_value(config, "qna.validator.timeout_s", 2.0))
    validator_url = get_config_value(
        config, "qna.validator.url", "http://localhost:11434"
    )
    validator = LLMValidator(
        model=validator_model, timeout_s=validator_timeout, url=validator_url
    )

    # Determine run_id (always timestamp-based)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    logger.info(f"- started_at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"- run_id: {run_id}(AS current style)")
    logger.info("CONFIG SUMMARY")
    logger.info(f"- model: {get_config_value(config, 'qna.llm.provider', 'ollama')}")
    logger.info(f"- validator: {validator_model}")
    logger.info(
        f"- run.time_limit_s: {get_config_value(config, 'qna.run.time_limit_s', 60)}"
    )
    logger.info(
        f"- run.llm_timeout_s: {get_config_value(config, 'qna.run.llm_timeout_s', 30)}"
    )
    logger.info(f"- run.max_turns: {get_config_value(config, 'qna.run.max_turns', 5)}")
    logger.info(f"- obs.mode: {get_config_value(config, 'qna.obs.mode', 'header_5')}")
    logger.info(
        f"- selection.mode: {get_config_value(config, 'qna.selection.mode', 'all')}"
    )
    logger.info(
        f"- selection.custom_ids: {get_config_value(config, 'qna.selection.custom_ids', [])}"
    )
    logger.info(f"- selection.batch_size: {batch_size}")
    logger.info(f"- selection.continue: {continue_mode}")
    if continue_from is not None:
        logger.info(f"- selection.continue_from: {continue_from}")
    # Integrate extractor logic: select and (optionally) limit records
    sel_cfg = SelectionConfig(
        mode=sel_mode, root=questions_path, custom_specs=custom_ids, seed=seed
    )
    t0 = time.perf_counter()
    records = collect_question_records(sel_cfg)
    t_select_s = time.perf_counter() - t0
    if limit is not None:
        try:
            n = int(limit)
            records = records[:n]
        except Exception:
            pass

    # Handle batching and continue logic
    batches = None
    all_records = None  # Initialize for batch info saving
    continue_batches = args.continue_batches or get_config_value(
        config, "qna.selection.continue_batches", False
    )
    logger.info(f"- selection.continue_batches: {continue_batches}")

    # Store original total for progress tracking
    original_total = len(records)
    continue_start_idx = 0  # Track where we started in the original list

    if continue_mode and continue_from is not None:
        # Continue mode: read previous batch_info and recreate the shuffled order
        continue_idx = None

        # Look for previous batch_info files to find the question position
        result_root = Path(
            get_config_value(config, "qna.result.root", "experiments/results")
        )
        batch_info_files = list(result_root.glob("*/batch_info_*.txt"))

        if batch_info_files:
            # Sort by modification time, get the most recent
            latest_batch_info = max(batch_info_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"[QnA] Reading batch info from: {latest_batch_info}")

            # Parse batch info to recreate the shuffled order
            with open(latest_batch_info, "r") as f:
                content = f.read()

            # Extract all questions from batch info in order
            shuffled_questions = []
            lines = content.split("\n")

            for line in lines:
                if line.strip() and line.strip()[0].isdigit() and "." in line:
                    # Extract question ID (format: "  7. 217")
                    parts = line.strip().split(".")
                    if len(parts) >= 2:
                        question_id = int(parts[1].strip())
                        shuffled_questions.append(question_id)

            logger.info(
                f"[QnA] Recreated shuffled order with {len(shuffled_questions)} questions"
            )

            # Find the question in the shuffled order
            for i, question_id in enumerate(shuffled_questions):
                if question_id == continue_from:
                    continue_idx = i
                    break

        if continue_idx is not None:
            # Recreate the same shuffle order using the same seed
            import random

            random.seed(seed)
            shuffled_records = records.copy()
            random.shuffle(shuffled_records)

            continue_start_idx = continue_idx
            records = shuffled_records[continue_idx:]
            logger.info(
                f"[QnA] Continuing from question {continue_from} (shuffled index {continue_idx})"
            )
            logger.info(f"[QnA] Continue mode: using recreated shuffle order")
        else:
            logger.warning(
                f"[QnA] Question {continue_from} not found in previous batch info, running all questions"
            )
    elif batch_size > 0 and len(records) > batch_size:
        # Split into batches
        import random

        random.seed(seed)
        all_records = records.copy()  # Keep original order for batch info
        random.shuffle(records)
        batches = [
            records[i : i + batch_size] for i in range(0, len(records), batch_size)
        ]
        logger.info(f"[QnA] Split into {len(batches)} batches of size {batch_size}")
        logger.info(f"[QnA] Batch 1: {len(batches[0])} questions")
        if continue_batches:
            logger.info(
                f"[QnA] Continue batches mode: will run all {len(batches)} batches"
            )
            records = records  # Run all records (all batches)
        else:
            logger.info(f"[QnA] Single batch mode: will run only batch 1")
            records = batches[0]  # Run only the first batch

    # Ensure model pulled/available
    _ensure_model_pulled(model_name, llm_url)

    # Print a brief summary and timing stats (selection elapsed)
    logger.info(
        f"[QnA] selected={len(records)} selection_time_s={t_select_s:.4f} sample={records[:2]}"
    )

    # Initialize observation space generator
    obs_generator = ObservationSpaceGenerator(config)

    # Prepare LLM client
    llm_opts = {
        "temperature": float(get_config_value(config, "qna.llm.temperature", 0.0)),
        "top_p": float(get_config_value(config, "qna.llm.top_p", 0.9)),
        "num_predict": int(get_config_value(config, "qna.llm.max_tokens", 512)),
    }
    provider = get_config_value(config, "qna.llm.provider", "ollama").lower()
    # Map provider → LLMClient backend
    if provider == "vllm":
        backend = "openai_compat"
    else:
        backend = "ollama"
    api_key = get_config_value(config, "qna.llm.api_key", None)
    client = LLMClient(
        model_name=model_name, url=llm_url, backend=backend, api_key=api_key
    )
    run_time_limit_s = float(get_config_value(config, "qna.run.time_limit_s", 60))
    llm_timeout_s = float(get_config_value(config, "qna.run.llm_timeout_s", 30))
    run_start = time.perf_counter()
    # Loop config merged into run
    max_turns = int(get_config_value(config, "qna.run.max_turns", 5))
    history_limit_cfg = get_config_value(config, "qna.run.history_limit", 5)
    repeats = int(get_config_value(config, "qna.run.repeats", 1))
    # logging configured above

    # Storage: create a per-run directory under result_root (always timestamp-based)
    run_dir = result_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save batch information if we have batches
    if batches is not None and all_records is not None:
        batch_info_file = run_dir / f"batch_info_{run_id}.txt"
        with open(batch_info_file, "w", encoding="utf-8") as f:
            f.write(f"Batch Information for Run {run_id}\n")
            f.write(f"Total questions: {len(all_records)}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Number of batches: {len(batches)}\n")
            f.write(f"Seed: {seed}\n\n")
            for i, batch in enumerate(batches, 1):
                f.write(f"Batch {i} ({len(batch)} questions):\n")
                for j, record in enumerate(batch, 1):
                    question_id = record.get("question_id", "unknown")
                    f.write(f"  {j}. {question_id}\n")
                f.write("\n")
        logger.info(f"[QnA] Batch information saved to {batch_info_file}")

    # Track consecutive Ollama timeouts
    ollama_timeout_count = 0
    max_ollama_timeouts = 3

    # Iterate over up to 'limit' records (already sliced above if qna.limit is set)
    for idx, test_record in enumerate(records):
        # Calculate correct progress: current position in original list
        current_progress = continue_start_idx + idx + 1
        logger.info("QUESTION PROGRESS")
        logger.info(f"- progress: {current_progress}/{original_total}")
        # observation space
        t_obs_start = time.perf_counter()
        obs_data = obs_generator.generate_obs_for_question(test_record)
        t_obs_s = time.perf_counter() - t_obs_start

        # prompt context
        # Normalize question id as question_XXX (001-based if numeric)
        raw_qid = test_record.get("question_id", "unknown")
        try:
            qid_num = int(raw_qid)
            qid_str = f"question_{qid_num:03d}"
        except Exception:
            qid_str = str(raw_qid)
        prompt_context = {
            "question_id": qid_str,
            "table_refs": test_record.get("table_refs", []),
            "question": test_record.get("question", ""),
            "answer": test_record.get("answer", ""),
            "answer_type": "numeric",
            "visible_columns_desc": obs_data["visible_columns_desc"],
            "visible_rows_desc": obs_data["visible_rows_desc"],
            "table_stats_desc": obs_data["table_stats_desc"],
            "exemplars_block": "",
        }
        logger.info(f"    - question_id: {qid_str}")
        logger.info(f"    - question: {prompt_context['question']}")

        t_prompt_start = time.perf_counter()
        try:
            full_prompt = prompt_manager.get_qna_prompt(prompt_context)
            t_prompt_s = time.perf_counter() - t_prompt_start
            logger.info(
                f"    - obs_time_s: {t_obs_s:.4f}  prompt_time_s: {t_prompt_s:.4f}"
            )
            logger.info(f"    - prompt_len: {len(full_prompt)}")
            logger.debug("-" * 50)
            logger.debug(
                full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt
            )
            logger.debug("-" * 50)
        except Exception as e:
            logger.error(f"[QnA] Error generating prompt: {e}")
            full_prompt = None

        # multi-round entry (we default to multi mode)
        if full_prompt is not None:
            # per-question section header for calls
            logger.info("    LLM CALLS")
            # repeats loop per question
            runs: list[dict] = []
            for r in range(1, repeats + 1):
                logger.info(f"  - repeat: {r}/{repeats}")
                t_question_start = time.perf_counter()
                (
                    final_answer,
                    _,
                    history,
                    end_reached,
                    validation_info,
                    question_timeout_count,
                ) = _multi_round_loop(
                    client=client,
                    base_prompt=full_prompt,
                    llm_opts=llm_opts,
                    llm_timeout_s=llm_timeout_s,
                    run_start=time.perf_counter(),
                    run_time_limit_s=run_time_limit_s,
                    max_turns=max_turns,
                    end_token="END",
                    include_prev_rounds=history_limit_cfg is not None,
                    history_limit=history_limit_cfg,
                    run_dir=str(run_dir),
                    question_text=prompt_context["question"],
                    golden_answer=prompt_context["answer"],
                    model_name=model_name,
                    question_id_str=qid_str,
                    validator=validator,
                    prompt_context=prompt_context,
                    config=config,
                )

                # Track consecutive Ollama timeouts
                if question_timeout_count > 0:
                    ollama_timeout_count += question_timeout_count
                else:
                    ollama_timeout_count = 0  # Reset counter on successful question

                # Check for consecutive Ollama timeouts
                if ollama_timeout_count >= max_ollama_timeouts:
                    logger.error("=" * 60)
                    logger.error("OLLAMA SERVER ERROR DETECTED")
                    logger.error("=" * 60)
                    logger.error(f"Consecutive Ollama timeouts: {ollama_timeout_count}")
                    logger.error(
                        "This indicates Ollama server issues, not complex questions."
                    )
                    logger.error("")
                    logger.error("TO CONTINUE FROM HERE:")
                    logger.error(f"1. Restart Ollama: ollama serve &")
                    logger.error(
                        f"2. Run with: --config configs/qna.yaml --continue --continue-from {idx + 2}"
                    )
                    logger.error(
                        f"   (Continue from question {idx + 2}, which is the next question)"
                    )
                    logger.error("")
                    logger.error("The script is stopping to prevent further timeouts.")
                    logger.error("=" * 60)
                    return 1
                t_question_wall = time.perf_counter() - t_question_start
                logger.info(f"    - total_elapsed_s: {t_question_wall:.3f}")
                run_entry = {
                    "steps": history,
                    "llm_answer": final_answer if end_reached else None,
                    "llm_valid": validation_info,
                }
                if not end_reached:
                    run_entry["error"] = "aborted_no_end"
                runs.append(run_entry)

            # write single file per question aggregating runs
            out_path = Path(run_dir) / f"{qid_str or 'unknown'}.jsonl"
            rec = {
                "question_id": qid_str,
                "question": prompt_context["question"],
                "golden_answer": prompt_context["answer"],
                "runs": runs,
            }
            write_json_pretty(out_path, rec)

    logger.info("[QnA] Run complete. See per-run outputs for details.")

    return 0


def _multi_round_loop(
    *,
    client: LLMClient,
    base_prompt: str,
    llm_opts: dict,
    llm_timeout_s: float,
    run_start: float,
    run_time_limit_s: float,
    max_turns: int,
    end_token: str,
    include_prev_rounds: bool,
    history_limit: int | str | None,
    run_dir: str,
    question_text: str | None,
    golden_answer: str | None,
    question_id_str: str,
    model_name: str,
    validator: LLMValidator,
    prompt_context: dict,
    config: dict,
) -> tuple[str | None, float | None, list[dict], bool, dict | None, int]:
    """Multi-round controller. Parses PLAN/END, validates/executes PLAN (DSL), logs decisions.
    Returns (final_text, total_llm_time_s, history, success, validation_info, timeout_count).
    """
    from src.utils.logging_utils import get_logger as _get_logger

    logger = _get_logger("qna_runner")
    total_llm_time = 0.0
    round_idx = 0
    prev_summary: list[str] = []
    history: list[dict] = []  # oldest first
    timeout_count = 0  # Track consecutive timeouts

    # Track repeated PLAN expressions for early termination
    repeated_plan_threshold = int(
        get_config_value(config, "qna.run.repeated_plan_threshold", 3)
    )
    plan_expression_count: dict[str, int] = {}  # plan_body -> count

    # Track failed PLAN expressions to detect stubborn behavior
    stubborn_threshold = int(get_config_value(config, "qna.run.stubborn_threshold", 3))
    failed_plan_count: dict[str, int] = {}  # plan_body -> failure count

    # Track stubborn behavior for forced termination
    stubborn_termination_threshold = int(
        get_config_value(config, "qna.run.stubborn_termination_threshold", 5)
    )

    def handle_error_with_stubborn_detection(content: str, code: str) -> str:
        """Handle error with stubborn behavior detection"""
        # Track failed content for stubborn behavior detection
        failed_plan_count[content] = failed_plan_count.get(content, 0) + 1
        failure_count = failed_plan_count[content]

        # Use guidance template with failure count for stubborn behavior detection
        guidance = _guidance_for(code, failure_count)

        if failure_count >= stubborn_threshold:
            logger.info(
                f"        - stubborn_behavior: true (failed {failure_count} times)"
            )

        return guidance

    def remaining_budget() -> float:
        return run_time_limit_s - (time.perf_counter() - run_start)

    def build_prompt_len_and_dump_chat() -> int:
        # Build chat messages only (single source of truth). Include history if configured.
        if include_prev_rounds and history:
            if history_limit in (None, "all"):
                hist_slice = history
            else:
                try:
                    k = int(history_limit)
                    hist_slice = history[-k:]
                except Exception:
                    hist_slice = history[-5:]
        else:
            hist_slice = []

        # Build exemplars according to config
        few_k = int(get_config_value(config, "qna.prompt.few_shot_k", 3))
        inc_cot = bool(get_config_value(config, "qna.prompt.include_cot", True))
        exemplars = prompt_manager.build_exemplars(
            few_shot_k=few_k, include_cot=inc_cot
        )
        # Check if this is the final round
        is_final_round = round_idx >= max_turns

        pm_context = {
            **prompt_context,
            "history": [
                {
                    "content": h.get("content", ""),
                    "status": h.get("status", ""),
                    "result_or_guidance": h.get("guidance")
                    or h.get("result")
                    or h.get("error", ""),
                }
                for h in hist_slice
            ],
            "exemplars_block": exemplars,
            "is_final_round": is_final_round,
        }
        messages = prompt_manager.get_chat_messages(pm_context)
        # TEMP dump chat messages if enabled
        if bool(get_config_value(config, "qna.prompt.dump_prompt", False)):
            dump_dir = Path(run_dir) / "prompt_dumps"
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_file = dump_dir / f"{question_id_str}_round{round_idx}_chat.txt"
            with open(dump_file, "w", encoding="utf-8") as f:
                for i, msg in enumerate(messages, 1):
                    f.write(
                        f"Message {i} ({msg['role']}):\n{'='*50}\n{msg['content']}\n\n"
                    )
        # return combined length for logging
        return sum(len(m.get("content", "")) for m in messages)

    while round_idx < max_turns:
        round_idx += 1
        if remaining_budget() <= 0:
            raise TimeoutError("TIME OUT【run】")
        prompt_len = build_prompt_len_and_dump_chat()
        logger.info(f"    - call: {round_idx}/{max_turns}")
        logger.info(f"        prompt_len: {prompt_len}")

        # Call LLM
        t0 = time.perf_counter()
        try:
            # Use chat API
            few_k = int(get_config_value(config, "qna.prompt.few_shot_k", 3))
            inc_cot = bool(get_config_value(config, "qna.prompt.include_cot", True))
            exemplars = prompt_manager.build_exemplars(
                few_shot_k=few_k, include_cot=inc_cot
            )
            pm_context = {
                **prompt_context,
                "history": [
                    {
                        "content": h.get("content", ""),
                        "status": h.get("status", ""),
                        "result_or_guidance": h.get("guidance")
                        or h.get("result")
                        or h.get("error", ""),
                    }
                    for h in (history if include_prev_rounds else [])
                ],
                "exemplars_block": exemplars,
            }
            messages = prompt_manager.get_chat_messages(pm_context)
            # chat dump handled in build_prompt_len_and_dump_chat for consistency
            resp_text = client.generate_chat(
                messages,
                options=llm_opts,
                timeout_s=min(llm_timeout_s, max(1.0, remaining_budget())),
            )
        except TimeoutError:
            code = "LLM_TIMEOUT"
            timeout_count += 1
            guidance = _guidance_for(code)
            logger.info("        status: ERROR")
            logger.info(f"        error_code: {code}")
            logger.info(f"        consecutive_timeouts: {timeout_count}")
            prev_summary.append(f"ERR {code}")
            history.append(
                {"content": "", "status": "FAILED", "error": code, "guidance": guidance}
            )
            continue
        except Exception as e:  # generic connection/client error
            code = "LLM_UNAVAILABLE"
            guidance = _guidance_for(code)
            logger.info("        status: ERROR")
            logger.info(f"        error_code: {code}")
            prev_summary.append(f"ERR {code}")
            history.append(
                {"content": "", "status": "FAILED", "error": code, "guidance": guidance}
            )
            continue
        t1 = time.perf_counter()
        dt = t1 - t0
        total_llm_time += dt
        logger.info(f"        call_time_s: {dt:.3f}")
        if resp_text:
            logger.debug(
                "        returned: \n"
                + resp_text[:800]
                + ("..." if len(resp_text) > 800 else "")
            )

        # Parse PLAN/END blocks
        # Check if current PLAN has been executed in this conversation
        # History stores DSL code, not "PLAN" strings, so check for successful execution
        plan_executed = any(
            step.get("status") == "SUCCESS" and step.get("result") is not None
            for step in history
        )
        logger.debug(f"        plan_executed: {plan_executed}")
        kind, content_or_err = parse_plan_or_end(resp_text or "", plan_executed)
        logger.debug(f"        parsed_kind: {kind}")

        # Check for repeated PLAN before processing
        if kind == "PLAN":
            plan_body = content_or_err
            # Check if this exact PLAN has been executed before
            if plan_body in plan_expression_count:
                current_count = plan_expression_count[plan_body] + 1
                plan_expression_count[plan_body] = current_count
                logger.debug(f"        repeated_plan_count: {current_count}")

                # If this PLAN has been repeated enough times, check if we have END block
                if current_count >= repeated_plan_threshold:
                    # Check if we already have an END block in history
                    has_end_block = False
                    for step in history:
                        if step.get("content", "").startswith("END"):
                            has_end_block = True
                            break

                    if not has_end_block:
                        # No END block found, use repeated PLAN detection
                        logger.info(
                            "    REPEATED PLAN DETECTION: No END block found, using repeated PLAN execution result"
                        )
                        # This will be handled later in the execution section
                    else:
                        # END block found, extract it instead of executing PLAN
                        logger.info(
                            "    REPEATED PLAN DETECTION: END block found, extracting END instead of executing PLAN"
                        )
                        # Change kind to END and extract the END content
                        kind = "END"
                        # Find the END block in the response
                        end_patterns = [
                            r"\*\*END\*\*\s*\n([\s\S]*?)(?:\n\*\*|$)",  # **END** followed by content
                            r"\*\*END\*\*\s+([^\n]+)(?:\n|$)",  # **END** followed by single line
                            r"(?:^|\n)END\s*\n([\s\S]*?)\nEND(?:\n|$)",  # Standard END block
                            r"(?:^|\n)END\s*\n([\s\S]*?)$",  # END at end of text
                            r"(?:^|\n)END\s+([^\n]+)(?:\n|$)",  # END followed by single line
                        ]

                        for pattern in end_patterns:
                            end_matches = re.findall(
                                pattern, resp_text, flags=re.IGNORECASE
                            )
                            if end_matches:
                                end_content = end_matches[0].strip()
                                if end_content and end_content != "PLAN":
                                    content_or_err = end_content
                                    break
            else:
                # New PLAN, initialize count
                plan_expression_count[plan_body] = 1
                logger.debug(f"        new_plan_count: 1")
        if kind == "ERROR":
            code = content_or_err
            guidance = _guidance_for(code)
            logger.info("        status: ERROR")
            logger.info(f"        error_code: {code}")
            prev_summary.append(f"FORMAT {code}")
            history.append(
                {"content": "", "status": "FAILED", "error": code, "guidance": guidance}
            )
            continue

        if kind == "PLAN":
            plan_body = content_or_err
            # Validate PLAN syntax
            try:
                table_refs = prompt_context.get("table_refs", [])
                _ = parse_plan(plan_body, table_refs)
            except Exception as e:
                code = "PLAN_SYNTAX"
                guidance = handle_error_with_stubborn_detection(plan_body, code)

                # Check for stubborn termination
                failure_count = failed_plan_count.get(plan_body, 0)
                if failure_count >= stubborn_termination_threshold:
                    logger.info(
                        "        - stubborn_termination: true (failed 5+ times)"
                    )
                    logger.info("        status: ERROR")
                    logger.info(f"        error_code: STUBBORN_TERMINATION")
                    prev_summary.append("STUBBORN_TERMINATION")
                    history.append(
                        {
                            "content": plan_body,
                            "status": "FAILED",
                            "error": "STUBBORN_TERMINATION",
                            "guidance": "Model repeatedly output the same incorrect PLAN formula. Forced termination.",
                        }
                    )
                    # Return error result
                    return (
                        None,
                        total_llm_time,
                        history,
                        False,
                        {
                            "decision": False,
                            "time_s": 0.0,
                            "retried": False,
                            "error": "STUBBORN_TERMINATION",
                        },
                        timeout_count,
                    )

                logger.info("        status: ERROR")
                logger.info(f"        error_code: {code}")
                prev_summary.append(code)
                history.append(
                    {
                        "content": plan_body,
                        "status": "FAILED",
                        "error": code,
                        "guidance": guidance,
                    }
                )
                continue

            # Check for repeated PLAN expressions
            plan_expression_count[plan_body] = (
                plan_expression_count.get(plan_body, 0) + 1
            )
            current_count = plan_expression_count[plan_body]

            # Execute PLAN (DSL)
            try:
                table_refs = prompt_context.get("table_refs", [])
                exec_df = execute_plan(plan_body, table_refs)
                logger.info("        status: OK")
                prev_summary.append("EXEC_OK")

                # Show COMPLETE results to model, not just preview
                if len(exec_df) <= 100:  # For small results, show everything
                    complete_result = exec_df.to_dict(orient="records")
                    logger.debug(
                        f"        complete_result: {len(complete_result)} rows"
                    )
                else:  # For large results, show all but log the count
                    complete_result = exec_df.to_dict(orient="records")
                    logger.debug(
                        f"        complete_result: {len(complete_result)} rows (large dataset)"
                    )

                history.append(
                    {
                        "content": plan_body,
                        "status": "SUCCESS",
                        "result": {"rows": len(exec_df), "preview": complete_result},
                    }
                )

                # Check if this PLAN has been executed enough times to trigger early termination
                # BUT only if we haven't already reached an END block
                if current_count >= repeated_plan_threshold:
                    # First check if we already have an END block in history
                    has_end_block = False
                    for step in history:
                        if step.get("content", "").startswith("END"):
                            has_end_block = True
                            break

                    if not has_end_block:
                        # Extract the result and use it as final answer
                        if complete_result and len(complete_result) > 0:
                            result_value = complete_result[0]
                            if (
                                isinstance(result_value, dict)
                                and len(result_value) == 1
                            ):
                                final_answer = str(list(result_value.values())[0])

                            logger.info(
                                "    REPEATED PLAN DETECTION: Using repeated PLAN execution result as final answer"
                            )
                            logger.info(f"    - extracted_answer: {final_answer}")
                            logger.info(
                                f"    - reached_END: false (repeated-generated)"
                            )
                            logger.info(f"    - plan_repeated_count: {current_count}")
                            logger.info(f"    - total_calls: {round_idx}")

                            # Use real validator
                            logger.info("    VALIDATION")
                            logger.info(f"    - golden_answer: {golden_answer}")
                            logger.info(f"    - llm_answer: {final_answer}")
                            logger.info(f"    - validator_model: {validator.model}")

                            validation_result = validator.validate(
                                question=question_text or "",
                                gold_answer=(
                                    str(golden_answer)
                                    if golden_answer is not None
                                    else ""
                                ),
                                model_answer=final_answer,
                            )

                            logger.info(
                                f"    - validator_decision: {validation_result.decision}"
                            )
                            logger.info(
                                f"    - validation_time_s: {validation_result.time_s:.3f}"
                            )
                            if validation_result.retried:
                                logger.info(f"    - validator_retried: true")
                            if validation_result.error:
                                logger.info(
                                    f"    - validator_error: {validation_result.error}"
                                )

                            # Add a final step to history indicating repeated-generated answer
                            history.append(
                                {
                                    "content": f"END {final_answer}",
                                    "status": "SUCCESS",
                                    "result": "final answer repeated-generated from repeated PLAN execution",
                                }
                            )

                            validation_info = {
                                "decision": validation_result.decision,
                                "time_s": validation_result.time_s,
                                "retried": validation_result.retried,
                                "error": validation_result.error,
                            }
                            return (
                                final_answer,
                                total_llm_time,
                                history,
                                True,
                                validation_info,
                                timeout_count,
                            )

                # After execution, proceed to next round for END
                continue
            except Exception as e:
                code = "EXEC_ERROR"
                guidance = handle_error_with_stubborn_detection(plan_body, code)

                # Check for stubborn termination
                failure_count = failed_plan_count.get(plan_body, 0)
                if failure_count >= stubborn_termination_threshold:
                    logger.info(
                        "        - stubborn_termination: true (failed 5+ times)"
                    )
                    logger.info("        status: ERROR")
                    logger.info(f"        error_code: STUBBORN_TERMINATION")
                    prev_summary.append("STUBBORN_TERMINATION")
                    history.append(
                        {
                            "content": plan_body,
                            "status": "FAILED",
                            "error": "STUBBORN_TERMINATION",
                            "guidance": "Model repeatedly output the same incorrect PLAN formula. Forced termination.",
                        }
                    )
                    # Return error result
                    return (
                        None,
                        total_llm_time,
                        history,
                        False,
                        {
                            "decision": False,
                            "time_s": 0.0,
                            "retried": False,
                            "error": "STUBBORN_TERMINATION",
                        },
                        timeout_count,
                    )

                logger.info("        status: ERROR")
                logger.info(f"        error_code: {code}")
                prev_summary.append(code)
                history.append(
                    {
                        "content": plan_body,
                        "status": "FAILED",
                        "error": code,
                        "guidance": guidance,
                    }
                )
                continue

        if kind == "END":
            final_answer = content_or_err.strip()

            # If END block is empty, try to extract answer from last PLAN execution result
            if not final_answer:
                logger.info(
                    "    END block is empty, extracting from last PLAN execution result"
                )
                for step in reversed(history):
                    if (
                        step.get("status") == "SUCCESS"
                        and step.get("result")
                        and isinstance(step.get("result"), dict)
                        and "preview" in step.get("result", {})
                    ):

                        # Extract the execution result
                        exec_result = step["result"]["preview"]
                        if exec_result and len(exec_result) > 0:
                            # For list questions, extract all unique values
                            unique_values = set()
                            for row in exec_result:
                                if isinstance(row, dict):
                                    for value in row.values():
                                        if value:
                                            unique_values.add(str(value))

                            if unique_values:
                                final_answer = ", ".join(sorted(unique_values))
                                logger.info(
                                    f"    - extracted {len(unique_values)} unique values from execution result"
                                )
                                break

            # Check if this END was found in the middle of a PLAN (abnormal case)
            is_abnormal_end = "PLAN" in resp_text and "END" in resp_text

            logger.info("    AGENT FINAL OUTPUT")
            logger.info(f"    - final_answer: {final_answer}")
            logger.info(f"    - reached_END: true")
            logger.info(f"    - total_calls: {round_idx}")
            if is_abnormal_end:
                logger.info(f"    - abnormal_end: true (END found within PLAN)")

            # Use real validator
            logger.info("    VALIDATION")
            logger.info(f"    - golden_answer: {golden_answer}")
            logger.info(f"    - llm_answer: {final_answer}")
            logger.info(f"    - validator_model: {validator.model}")

            validation_result = validator.validate(
                question=question_text or "",
                gold_answer=str(golden_answer) if golden_answer is not None else "",
                model_answer=final_answer,
            )

            logger.info(f"    - validator_decision: {validation_result.decision}")
            logger.info(f"    - validation_time_s: {validation_result.time_s:.3f}")
            if validation_result.retried:
                logger.info(f"    - validator_retried: true")
            if validation_result.error:
                logger.info(f"    - validator_error: {validation_result.error}")

            # Mark the result type based on whether it's abnormal
            result_type = "abnormal_end" if is_abnormal_end else "normal_end"
            history.append(
                {
                    "content": f"END {final_answer}",
                    "status": "SUCCESS",
                    "result": f"final answer from {result_type} block",
                }
            )
            validation_info = {
                "decision": validation_result.decision,
                "time_s": validation_result.time_s,
                "retried": validation_result.retried,
                "error": validation_result.error,
            }
            return (
                final_answer,
                total_llm_time,
                history,
                True,
                validation_info,
                timeout_count,
            )

    logger.info("[loop] max_turns reached without END")

    # SPECIAL HANDLING: If the last round was a successful PLAN execution,
    # treat the execution result as the final answer
    if history and len(history) > 0:
        # Look for the last successful execution in history
        for step in reversed(history):
            if (
                step.get("status") == "SUCCESS"
                and step.get("result")
                and isinstance(step.get("result"), dict)
                and "preview" in step.get("result", {})
            ):

                # Extract the execution result
                exec_result = step["result"]["preview"]
                if exec_result and len(exec_result) > 0:
                    # Get the first (and likely only) result value
                    result_value = exec_result[0]
                    if isinstance(result_value, dict) and len(result_value) == 1:
                        # Extract the single value from the result
                        final_answer = str(list(result_value.values())[0])

                        logger.info(
                            "    SPECIAL HANDLING: Using last successful PLAN execution result as final answer"
                        )
                        logger.info(f"    - extracted_answer: {final_answer}")
                        logger.info(f"    - reached_END: false (auto-generated)")
                        logger.info(f"    - total_calls: {round_idx}")

                        # Use real validator
                        logger.info("    VALIDATION")
                        logger.info(f"    - golden_answer: {golden_answer}")
                        logger.info(f"    - llm_answer: {final_answer}")
                        logger.info(f"    - validator_model: {validator.model}")

                        validation_result = validator.validate(
                            question=question_text or "",
                            gold_answer=(
                                str(golden_answer) if golden_answer is not None else ""
                            ),
                            model_answer=final_answer,
                        )

                        logger.info(
                            f"    - validator_decision: {validation_result.decision}"
                        )
                        logger.info(
                            f"    - validation_time_s: {validation_result.time_s:.3f}"
                        )
                        if validation_result.retried:
                            logger.info(f"    - validator_retried: true")
                        if validation_result.error:
                            logger.info(
                                f"    - validator_error: {validation_result.error}"
                            )

                        # Add a final step to history indicating auto-generated answer
                        history.append(
                            {
                                "content": f"END {final_answer}",
                                "status": "SUCCESS",
                                "result": "final answer auto-generated from last successful PLAN execution",
                            }
                        )

                        validation_info = {
                            "decision": validation_result.decision,
                            "time_s": validation_result.time_s,
                            "retried": validation_result.retried,
                            "error": validation_result.error,
                        }
                        return (
                            final_answer,
                            total_llm_time,
                            history,
                            True,
                            validation_info,
                            timeout_count,
                        )
                break

    # Write an aborted record so the run directory contains a file even without END
    try:
        pass
    except Exception:
        pass
    return None, total_llm_time, history, False, None, timeout_count


## parser moved to src/evaluation/plan_parser.py


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
