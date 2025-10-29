#!/usr/bin/env python3
"""
Experiment Results Analysis Script

This script analyzes experiment results from multiple runs and generates
comprehensive analysis reports with visualizations.

Usage:
    python experiments/analysis/analyze_results.py --config configs/analysis.yaml
    python experiments/analysis/analyze_results.py --config configs/analysis.yaml --output-dir custom_analysis
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.config import load_config, get_config_value
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


class ExperimentAnalyzer:
    """Analyzes experiment results and generates comprehensive reports."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_config = get_config_value(config, "analysis", {})
        self.input_config = get_config_value(self.analysis_config, "input", {})
        self.output_config = get_config_value(self.analysis_config, "output", {})
        self.visualization_config = get_config_value(config, "visualizations", {})
        
        # Get statistics configuration
        self.statistics_config = get_config_value(self.analysis_config, "statistics", {})
        self.detailed_config = get_config_value(self.analysis_config, "detailed", {})
        self.output_formats_config = get_config_value(self.analysis_config, "output_formats", {})
        
        # Setup output directory
        self.output_dir = Path(self.output_config.get("output_dir", "experiments/analysis"))
        self.analysis_name = self.output_config.get("analysis_name", "analysis")
        self.analysis_path = self.output_dir / self.analysis_name
        # Clean existing output directory
        if self.analysis_path.exists():
            import shutil
            shutil.rmtree(self.analysis_path)
        self.analysis_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging (clear existing log file)
        log_file = self.analysis_path / f"{self.analysis_name}.log"
        if log_file.exists():
            log_file.unlink()  # Remove existing log file
        setup_logging(
            log_level="INFO",
            log_file=log_file,
            console_output=True
        )
        
        logger.info(f"Initialized analyzer for: {self.analysis_name}")
        logger.info(f"Output directory: {self.analysis_path}")
    
    def load_experiment_data(self) -> List[Dict[str, Any]]:
        """Load experiment data from specified result directories."""
        result_dirs = self.input_config.get("result_dirs", [])
        question_ids = self.input_config.get("question_ids", [])
        question_folder = self.input_config.get("question_folder", "data/questions")
        all_data = []
        
        for result_dir in result_dirs:
            result_path = Path(result_dir)
            if not result_path.exists():
                logger.warning(f"Result directory not found: {result_dir}")
                continue
            
            logger.info(f"Loading data from: {result_dir}")
            
            # Load JSONL files
            jsonl_files = list(result_path.glob("question_*.jsonl"))
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Try to parse as single JSON object first
                            try:
                                data = json.loads(content)
                                data['source_dir'] = str(result_path)
                                data['source_file'] = jsonl_file.name
                                
                                # Load metadata from corresponding question.json file
                                metadata = self._load_question_metadata(data.get('question_id'), question_folder)
                                if metadata:
                                    data.update(metadata)
                                    logger.debug(f"Loaded metadata for {data.get('question_id')}: {list(metadata.keys())}")
                                
                                # Process multiple runs if present
                                if 'runs' in data:
                                    # Flatten runs into separate records
                                    for run_idx, run in enumerate(data['runs']):
                                        run_data = data.copy()
                                        run_data['run_index'] = run_idx
                                        run_data['total_runs'] = len(data['runs'])
                                        
                                        # Extract run-specific information
                                        if 'steps' in run:
                                            run_data['steps'] = run['steps']
                                            # Get final status from last step
                                            if run['steps']:
                                                last_step = run['steps'][-1]
                                                run_data['status'] = last_step.get('status', 'UNKNOWN')
                                                run_data['error_code'] = last_step.get('error_code')
                                        
                                        # Extract LLM validation information
                                        if 'llm_valid' in run:
                                            run_data['llm_valid'] = run['llm_valid']
                                        if 'llm_answer' in run:
                                            run_data['llm_answer'] = run['llm_answer']
                                        if 'error' in run:
                                            run_data['error'] = run['error']
                                        
                                        all_data.append(run_data)
                                else:
                                    all_data.append(data)
                            except json.JSONDecodeError:
                                # If single JSON fails, try line-by-line JSONL format
                                f.seek(0)
                                for line_num, line in enumerate(f, 1):
                                    if line.strip():
                                        try:
                                            data = json.loads(line.strip())
                                            data['source_dir'] = str(result_path)
                                            data['source_file'] = jsonl_file.name
                                            
                                            # Load metadata from corresponding question.json file
                                            metadata = self._load_question_metadata(data.get('question_id'), question_folder)
                                            if metadata:
                                                data.update(metadata)
                                            
                                            all_data.append(data)
                                        except json.JSONDecodeError as e:
                                            logger.error(f"JSON decode error in {jsonl_file}:{line_num}: {e}")
                except Exception as e:
                    logger.error(f"Error loading {jsonl_file}: {e}")
        
        # Filter by question_ids if specified
        if question_ids:
            original_count = len(all_data)
            # Convert question_ids to string format for comparison
            question_id_strings = [f"question_{qid:03d}" for qid in question_ids]
            all_data = [record for record in all_data if record.get('question_id') in question_id_strings]
            logger.info(f"Filtered to {len(all_data)} records (from {original_count}) for specified question_ids: {question_ids}")
        else:
            logger.info(f"Loaded {len(all_data)} experiment records")
        
        return all_data
    
    def _load_question_metadata(self, question_id: str, question_folder: str = "data/questions") -> Dict[str, Any]:
        """Load metadata from question.json file with universal path search."""
        if not question_id:
            return {}
        
        question_folder_path = Path(question_folder)
        if not question_folder_path.exists():
            logger.warning(f"Question folder not found: {question_folder}")
            return {}
        
        # Search for question.json file recursively in all subdirectories
        for json_file in question_folder_path.rglob(f"{question_id}.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    logger.debug(f"Found metadata file: {json_file}")
                    
                    # Extract relevant metadata fields
                    # Convert complexity score to categorical (based on table_distribution_tracker.md)
                    complexity_score = metadata.get('complexity_metrics', {}).get('complexity_score', 0)
                    if complexity_score <= 6.0:
                        complexity_level = 'low'
                    elif complexity_score <= 21.5:
                        complexity_level = 'medium'
                    else:
                        complexity_level = 'high'

                    return {
                        'reasoning_type': metadata.get('reasoning_type'),
                        'domain': metadata.get('domain'),
                        'table_number': metadata.get('table_number'),
                        'answer_type': metadata.get('answer_type'),
                        'answerable': metadata.get('answerable'),
                        'complexity': complexity_level,
                        'distractor_type': metadata.get('distractor_type'),
                        'requires_calculation': metadata.get('requires_calculation'),
                        'has_distractor': metadata.get('has_distractor')
                    }
            except Exception as e:
                logger.warning(f"Error loading metadata from {json_file}: {e}")
                continue
        
        logger.warning(f"No metadata file found for {question_id} in {question_folder}")
        return {}
    
    def generate_summary_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from experiment data."""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        # Basic statistics
        total_questions = len(df)
        unique_questions = df['question_id'].nunique() if 'question_id' in df.columns else 0
        
        # Extract LLM validation results
        df['llm_decision'] = None
        if 'llm_valid' in df.columns:
            # Extract decision from llm_valid column
            def extract_llm_decision(llm_valid_data):
                if isinstance(llm_valid_data, dict):
                    return llm_valid_data.get('decision')
                return None
            
            df['llm_decision'] = df['llm_valid'].apply(extract_llm_decision)
        
        # Success rate analysis based on LLM validation (only TRUE counts as success)
        success_rate = 0
        if 'llm_decision' in df.columns:
            success_count = len(df[df['llm_decision'] == True])
            success_rate = success_count / total_questions if total_questions > 0 else 0
        
        # Result distribution based on LLM validation
        result_distribution = {}
        if 'llm_decision' in df.columns:
            # Map True/False/None to readable labels
            df['result_label'] = df['llm_decision'].map({
                True: 'SUCCESS',
                False: 'FAILED', 
                None: 'NULL_ANSWER'
            })
            result_distribution = df['result_label'].value_counts().to_dict()
        
        # DSL execution error analysis (extract from all steps)
        dsl_error_distribution = {}
        all_step_statuses = []
        all_error_codes = []
        
        # Extract status and error from all steps in all runs
        for record in data:
            if 'steps' in record and isinstance(record['steps'], list):
                for step in record['steps']:
                    if isinstance(step, dict):
                        if 'status' in step:
                            all_step_statuses.append(step['status'])
                        # Only collect error if status is FAILED
                        if (step.get('status') == 'FAILED' and 
                            'error' in step and step['error']):
                            all_error_codes.append(step['error'])
        
        if all_step_statuses:
            dsl_error_distribution = pd.Series(all_step_statuses).value_counts().to_dict()
        
        # For pie chart, combine SUCCESS count with error codes
        dsl_error_codes = {}
        if all_step_statuses:
            status_counts = pd.Series(all_step_statuses).value_counts()
            # Add SUCCESS count
            if 'SUCCESS' in status_counts:
                dsl_error_codes['SUCCESS'] = status_counts['SUCCESS']
            # Add error codes from FAILED steps
            if all_error_codes:
                error_counts = pd.Series(all_error_codes).value_counts()
                dsl_error_codes.update(error_counts.to_dict())
        
        # LLM output round analysis (count LLM feedback rounds per run)
        llm_round_stats = {
            'total_llm_rounds': 0,
            'avg_rounds_per_run': 0,
            'rounds_per_run': []
        }
        
        # Additional statistics
        error_pattern_stats = {
            'most_common_errors': {},
            'error_frequency_by_question': {},
            'avg_errors_per_run': 0,
            'total_error_steps': 0
        }
        
        # NULL answer error analysis
        null_answer_errors = {}
        null_answer_count = 0
        
        for record in data:
            # Check for NULL answer errors
            if record.get('llm_answer') is None:
                null_answer_count += 1
                error_type = record.get('error', 'no_error_field')
                if error_type not in null_answer_errors:
                    null_answer_errors[error_type] = 0
                null_answer_errors[error_type] += 1
            
            if 'steps' in record and isinstance(record['steps'], list):
                # Count LLM output rounds (steps with 'content' that are LLM responses)
                llm_rounds = 0
                error_count = 0
                question_id = record.get('question_id', 'unknown')
                
                for step in record['steps']:
                    if isinstance(step, dict) and 'content' in step:
                        content = step.get('content', '')
                        # Count as LLM round if it contains LLM output (not just system messages)
                        if content and not content.startswith('System:') and not content.startswith('User:'):
                            llm_rounds += 1
                    
                    # Count errors
                    if step.get('status') == 'FAILED':
                        error_count += 1
                        error_type = step.get('error', 'UNKNOWN')
                        if error_type not in error_pattern_stats['most_common_errors']:
                            error_pattern_stats['most_common_errors'][error_type] = 0
                        error_pattern_stats['most_common_errors'][error_type] += 1
                
                llm_round_stats['total_llm_rounds'] += llm_rounds
                llm_round_stats['rounds_per_run'].append(llm_rounds)
                error_pattern_stats['total_error_steps'] += error_count
                
                # Track errors by question
                if question_id not in error_pattern_stats['error_frequency_by_question']:
                    error_pattern_stats['error_frequency_by_question'][question_id] = 0
                error_pattern_stats['error_frequency_by_question'][question_id] += error_count
        
        if llm_round_stats['rounds_per_run']:
            llm_round_stats['avg_rounds_per_run'] = sum(llm_round_stats['rounds_per_run']) / len(llm_round_stats['rounds_per_run'])
        
        if len(data) > 0:
            error_pattern_stats['avg_errors_per_run'] = error_pattern_stats['total_error_steps'] / len(data)
        
        # Time analysis
        time_stats = {}
        time_column = None
        if 'execution_time' in df.columns:
            time_column = 'execution_time'
        elif 'time_s' in df.columns:
            time_column = 'time_s'
        
        if time_column and not df[time_column].empty:
            time_stats = {
                'mean_time': df[time_column].mean(),
                'median_time': df[time_column].median(),
                'min_time': df[time_column].min(),
                'max_time': df[time_column].max()
            }
        
        # Metadata-based analysis (if enabled)
        metadata_analysis = {}
        if self.statistics_config.get('metadata_analysis', True):
            metadata_analysis = self._analyze_by_metadata(df)
        
        # Consistency analysis for multiple runs (if enabled)
        consistency_analysis = {}
        if self.statistics_config.get('consistency_analysis', True):
            consistency_analysis = self._analyze_consistency(data)
        
        return {
            'total_questions': total_questions,
            'unique_questions': unique_questions,
            'success_rate': success_rate,
            'result_distribution': result_distribution,
            'dsl_error_distribution': dsl_error_distribution,
            'dsl_error_codes': dsl_error_codes,
            'llm_round_stats': llm_round_stats,
            'error_pattern_stats': error_pattern_stats,
            'null_answer_errors': null_answer_errors,
            'null_answer_count': null_answer_count,
            'time_statistics': time_stats,
            'data_sources': df['source_dir'].unique().tolist() if 'source_dir' in df.columns else [],
            'metadata_analysis': metadata_analysis,
            'consistency_analysis': consistency_analysis
        }
    
    def generate_visualizations(self, data: List[Dict[str, Any]], stats: Dict[str, Any]):
        """Generate visualization charts."""
        if not data:
            logger.warning("No data available for visualization")
            return
        
        df = pd.DataFrame(data)
        
        # Extract LLM validation results for visualization
        df['llm_decision'] = None
        if 'llm_valid' in df.columns:
            def extract_llm_decision(llm_valid_data):
                if isinstance(llm_valid_data, dict):
                    return llm_valid_data.get('decision')
                return None
            
            df['llm_decision'] = df['llm_valid'].apply(extract_llm_decision)
        
        # Create result labels for visualization
        if 'llm_decision' in df.columns:
            df['result_label'] = df['llm_decision'].map({
                True: 'SUCCESS',
                False: 'FAILED', 
                None: 'NULL_ANSWER'
            })
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Check if visualizations are enabled
        if not self.output_formats_config.get('visualizations', True):
            logger.info("Visualizations disabled in configuration")
            return
        
        # Get enabled charts from configuration
        enabled_charts = self.visualization_config.get('charts', [])
        
        # 1. LLM Validation Results Distribution
        if 'result_label' in df.columns and ('results_distribution' in enabled_charts or not enabled_charts):
            plt.figure(figsize=(12, 8))
            result_counts = df['result_label'].value_counts()
            
            # Define colors for different result types
            colors = []
            labels = []
            for result in result_counts.index:
                if result == 'SUCCESS':
                    colors.append('#2ecc71')  # Green
                    labels.append('SUCCESS')
                elif result == 'FAILED':
                    colors.append('#e74c3c')  # Red
                    labels.append('FAILED')
                elif result == 'NULL_ANSWER':
                    colors.append('#f39c12')  # Orange
                    labels.append('NULL_ANSWER')
                else:
                    colors.append('#95a5a6')  # Gray
                    labels.append(result)
            
            # Create custom autopct function to show both count and percentage
            def autopct_format(pct):
                count = int(pct/100.*sum(result_counts.values))
                return f'{count}\n({pct:.1f}%)'
            
            plt.pie(result_counts.values, labels=labels, colors=colors, autopct=autopct_format, startangle=90)
            plt.title('LLM Validation Results Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            plt.savefig(self.analysis_path / 'results_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Generated LLM validation results distribution chart")
        
        # 2. DSL Execution Status Distribution (Bar Chart)
        dsl_errors = stats.get('dsl_error_distribution', {})
        if dsl_errors and ('dsl_execution_errors' in enabled_charts or not enabled_charts):
            plt.figure(figsize=(12, 6))
            statuses = list(dsl_errors.keys())
            counts = list(dsl_errors.values())
            total_count = sum(counts)
            
            bars = plt.bar(statuses, counts, color='lightcoral', edgecolor='darkred')
            plt.title('DSL Execution Status Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Execution Status', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45)
            
            # Adjust Y-axis range to give more space for labels
            max_count = max(counts) if counts else 1
            plt.ylim(0, max_count * 1.3)  # Add 30% more space above max value
            
            # Add value labels on bars (count and percentage)
            for bar, count in zip(bars, counts):
                percentage = (count / total_count) * 100 if total_count > 0 else 0
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(self.analysis_path / 'dsl_execution_errors.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Generated DSL execution error distribution chart")
        else:
            logger.debug("No DSL execution data available for visualization")
        
        # 3. DSL Execution Error Codes Pie Chart
        dsl_error_codes = stats.get('dsl_error_codes', {})
        if dsl_error_codes and ('dsl_execution_pie' in enabled_charts or not enabled_charts):
            plt.figure(figsize=(12, 8))
            error_codes = list(dsl_error_codes.keys())
            counts = list(dsl_error_codes.values())
            total_count = sum(counts)
            
            # Separate main categories from small categories
            main_categories = []
            main_counts = []
            main_colors = []
            main_labels = []
            small_categories = []
            small_counts = []
            
            color_map = {
                'SUCCESS': '#2ecc71',
                'EXEC_ERROR': '#e74c3c', 
                'PLAN_SYNTAX': '#f39c12',
                'FORMAT_NO_BLOCK': '#9b59b6'
            }
            
            for error_code, count in zip(error_codes, counts):
                percentage = (count / total_count) * 100
                if percentage >= 1.0:  # Show categories with >= 1% in main pie
                    main_categories.append(error_code)
                    main_counts.append(count)
                    main_colors.append(color_map.get(error_code, '#95a5a6'))
                    main_labels.append(error_code)
                else:  # Show categories with < 1% in corner box
                    small_categories.append(error_code)
                    small_counts.append(count)
            
            # Create custom autopct function to show both count and percentage
            def autopct_format(pct):
                count = int(pct/100.*sum(main_counts))
                return f'{count}\n({pct:.1f}%)'
            
            # Create main pie chart
            if main_counts:
                plt.pie(main_counts, labels=main_labels, colors=main_colors, 
                       autopct=autopct_format, startangle=90)
            
            # Add small categories in a text box in the corner
            if small_categories:
                small_text = "Minor Categories:\n"
                for code, count in zip(small_categories, small_counts):
                    percentage = (count / total_count) * 100
                    small_text += f"{code}: {count} ({percentage:.1f}%)\n"
                
                plt.text(0.02, 0.02, small_text, transform=plt.gca().transAxes, 
                        fontsize=9, verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
            
            plt.title('DSL Execution Error Codes Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(self.analysis_path / 'dsl_execution_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Generated DSL execution error codes pie chart")
        else:
            logger.debug("No DSL error codes available for pie chart")
        
        # 3.5. NULL Answer Errors Pie Chart
        null_answer_errors = stats.get('null_answer_errors', {})
        if null_answer_errors and ('null_answer_errors' in enabled_charts or not enabled_charts):
            plt.figure(figsize=(10, 8))
            error_types = list(null_answer_errors.keys())
            counts = list(null_answer_errors.values())
            total_count = sum(counts)
            
            # Define colors for different error types
            colors = ['#e74c3c', '#f39c12', '#9b59b6', '#34495e', '#95a5a6']
            
            # Create custom autopct function to show both count and percentage
            def autopct_format(pct):
                count = int(pct/100.*sum(counts))
                return f'{count}\n({pct:.1f}%)'
            
            plt.pie(counts, labels=error_types, colors=colors[:len(error_types)], 
                   autopct=autopct_format, startangle=90)
            plt.title('NULL Answer Error Types Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(self.analysis_path / 'null_answer_errors_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Generated NULL answer errors pie chart")
        else:
            logger.debug("No NULL answer errors available for pie chart")
        
        # 4. LLM Output Rounds Distribution
        llm_round_stats = stats.get('llm_round_stats', {})
        if (llm_round_stats and 'rounds_per_run' in llm_round_stats and 
            ('llm_rounds_distribution' in enabled_charts or not enabled_charts)):
            rounds_data = llm_round_stats['rounds_per_run']
            if rounds_data:
                plt.figure(figsize=(12, 6))
                rounds_counts = pd.Series(rounds_data).value_counts().sort_index()
                total_runs = len(rounds_data)
                
                # Use narrower bars and adjust positioning
                bar_width = 0.6
                x_positions = range(len(rounds_counts.index))
                bars = plt.bar(x_positions, rounds_counts.values, width=bar_width, 
                              color='lightblue', edgecolor='navy', align='center')
                
                plt.title('LLM Output Rounds per Run Distribution', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Number of LLM Rounds', fontsize=12)
                plt.ylabel('Number of Runs', fontsize=12)
                
                # Set x-axis ticks to be centered on bars
                plt.xticks(x_positions, rounds_counts.index)
                
                # Adjust Y-axis range to give more space for labels
                max_count = max(rounds_counts.values) if len(rounds_counts) > 0 else 1
                plt.ylim(0, max_count * 1.3)  # Add 30% more space above max value
                
                # Add value labels on bars (count and percentage)
                for bar, count in zip(bars, rounds_counts.values):
                    percentage = (count / total_runs) * 100 if total_runs > 0 else 0
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', 
                            fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'llm_rounds_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated LLM rounds distribution chart")
        
        # 2. Error Distribution (if available)
        if 'error_code' in df.columns:
            error_counts = df['error_code'].value_counts()
            if not error_counts.empty:
                plt.figure(figsize=(12, 6))
                error_counts.plot(kind='bar')
                plt.title('Error Code Distribution')
                plt.xlabel('Error Code')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'error_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated error distribution chart")
            else:
                logger.debug("No error codes available for visualization")
        
        # 5. Execution Time Distribution
        if ('execution_time_distribution' in enabled_charts or not enabled_charts):
            time_column = None
            if 'execution_time' in df.columns:
                time_column = 'execution_time'
            elif 'time_s' in df.columns:
                time_column = 'time_s'
            
            if time_column and time_column in df.columns and not df[time_column].empty:
                plt.figure(figsize=(12, 8))
                # Use more appropriate bins for execution time
                time_values = df[time_column].dropna()
                if len(time_values) > 0:
                    bins = max(10, min(30, len(time_values) // 2))
                    plt.hist(time_values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title('Execution Time Distribution', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Execution Time (seconds)', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                    
                    # Add statistics text
                    mean_time = time_values.mean()
                    median_time = time_values.median()
                    plt.axvline(mean_time, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_time:.2f}s')
                    plt.axvline(median_time, color='green', linestyle='--', alpha=0.7, label=f'Median: {median_time:.2f}s')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig(self.analysis_path / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("Generated execution time distribution chart")
        else:
            logger.debug("No time data available for visualization")
        
        logger.info(f"Generated visualizations in: {self.analysis_path}")
        
        # 4. Metadata-based visualizations
        if ('success_by_reasoning_type' in enabled_charts or 
            'success_by_complexity' in enabled_charts or 
            'success_by_domain' in enabled_charts or 
            'success_by_answerable' in enabled_charts or
            not enabled_charts):
            try:
                self._generate_metadata_visualizations(df, stats)
            except Exception as e:
                logger.error(f"Error in metadata visualizations: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
    
    def _generate_metadata_visualizations(self, df: pd.DataFrame, stats: Dict[str, Any]):
        """Generate visualizations for metadata dimensions."""
        metadata_analysis = stats.get('metadata_analysis', {})
        logger.debug(f"Metadata analysis keys: {list(metadata_analysis.keys())}")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        
        # Success rate by reasoning type
        if 'reasoning_type' in metadata_analysis and metadata_analysis['reasoning_type']:
            reasoning_data = metadata_analysis['reasoning_type'].get('success_rate', {})
            if reasoning_data and len(reasoning_data) > 0:
                plt.figure(figsize=(14, 8))
                reasoning_types = list(reasoning_data.keys())
                success_rates = list(reasoning_data.values())
                
                bars = plt.bar(reasoning_types, success_rates, color='lightblue', edgecolor='navy')
                plt.title('Success Rate by Reasoning Type', fontsize=18, fontweight='bold', pad=30)
                plt.xlabel('Reasoning Type', fontsize=14)
                plt.ylabel('Success Rate (%)', fontsize=14)
                plt.xticks(rotation=45, fontsize=12)
                plt.yticks(fontsize=12)
                
                # Dynamic Y-axis range for better differentiation
                max_rate = max(success_rates) if success_rates else 0
                y_max = min(1.0, max_rate * 1.2)  # 20% above max rate, but not exceed 100%
                plt.ylim(0, y_max)
                # Format y-axis as percentage
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                
                # Add value labels on bars with percentage
                for bar, rate in zip(bars, success_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                            f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'success_by_reasoning_type.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated success rate by reasoning type chart")
            else:
                logger.debug("No reasoning type data available for visualization")
        
        # Success rate by complexity
        if 'complexity' in metadata_analysis and metadata_analysis['complexity']:
            complexity_data = metadata_analysis['complexity'].get('success_rate', {})
            if complexity_data and len(complexity_data) > 0:
                plt.figure(figsize=(12, 8))
                # Sort by complexity level order
                complexity_order = ['low', 'medium', 'high']
                complexities = [c for c in complexity_order if c in complexity_data.keys()]
                success_rates = [complexity_data[c] for c in complexities]
                
                if complexities and success_rates:
                    bars = plt.bar(complexities, success_rates, color=['lightgreen', 'orange', 'red'], 
                                  edgecolor='black', linewidth=1.5)
                    plt.title('Success Rate by Complexity Level', fontsize=18, fontweight='bold', pad=30)
                    plt.xlabel('Complexity Level', fontsize=14)
                    plt.ylabel('Success Rate (%)', fontsize=14)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)
                    
                    # Dynamic Y-axis range for better differentiation
                    max_rate = max(success_rates) if success_rates else 0
                    y_max = min(1.0, max_rate * 1.2)  # 20% above max rate, but not exceed 100%
                    plt.ylim(0, y_max)
                    # Format y-axis as percentage
                    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                    
                    # Add value labels on bars with percentage
                    for bar, rate in zip(bars, success_rates):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                                f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.analysis_path / 'success_by_complexity.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info("Generated success rate by complexity chart")
                else:
                    logger.debug("No complexity data available for visualization")
            else:
                logger.debug("No complexity data available for visualization")
        
        # Success rate by domain
        if 'domain' in metadata_analysis and metadata_analysis['domain']:
            domain_data = metadata_analysis['domain'].get('success_rate', {})
            if domain_data and len(domain_data) > 0:
                plt.figure(figsize=(16, 10))
                domains = list(domain_data.keys())
                success_rates = list(domain_data.values())
                
                # Sort by success rate for better visualization
                sorted_data = sorted(zip(domains, success_rates), key=lambda x: x[1], reverse=True)
                domains, success_rates = zip(*sorted_data)
                
                bars = plt.bar(range(len(domains)), success_rates, color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
                plt.title('Success Rate by Domain', fontsize=18, fontweight='bold', pad=30)
                plt.xlabel('Domain', fontsize=14)
                plt.ylabel('Success Rate (%)', fontsize=14)
                plt.xticks(range(len(domains)), domains, rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                
                # Dynamic Y-axis range for better differentiation
                max_rate = max(success_rates) if success_rates else 0
                y_max = min(1.0, max_rate * 1.2)  # 20% above max rate, but not exceed 100%
                plt.ylim(0, y_max)
                # Format y-axis as percentage
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                
                # Add value labels on bars with percentage
                for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                            f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'success_by_domain.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated success rate by domain chart")
            else:
                logger.debug("No domain data available for visualization")
        
        # Success rate by has_distractor
        if 'has_distractor' in metadata_analysis and metadata_analysis['has_distractor']:
            has_distractor_data = metadata_analysis['has_distractor'].get('success_rate', {})
            if has_distractor_data and len(has_distractor_data) > 0:
                plt.figure(figsize=(10, 6))
                has_distractor_values = list(has_distractor_data.keys())
                success_rates = list(has_distractor_data.values())
                
                # Convert boolean values to string labels for better display
                labels = ['No Distractor' if x == False else 'Has Distractor' for x in has_distractor_values]
                
                bars = plt.bar(labels, success_rates, color=['lightcoral', 'lightblue'], 
                              edgecolor='black', linewidth=1.5)
                plt.title('Success Rate by Distractor Presence', fontsize=18, fontweight='bold', pad=30)
                plt.xlabel('Distractor Presence', fontsize=14)
                plt.ylabel('Success Rate (%)', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                
                # Dynamic Y-axis range for better differentiation
                max_rate = max(success_rates) if success_rates else 0
                y_max = min(1.0, max_rate * 1.2)  # 20% above max rate, but not exceed 100%
                plt.ylim(0, y_max)
                # Format y-axis as percentage
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                
                # Add value labels on bars with percentage
                for bar, rate in zip(bars, success_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                            f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'success_by_has_distractor.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated success rate by has_distractor chart")
            else:
                logger.debug("No has_distractor data available for visualization")
        
        # Success rate by answerable
        if 'answerable' in metadata_analysis and metadata_analysis['answerable']:
            answerable_data = metadata_analysis['answerable'].get('success_rate', {})
            if answerable_data and len(answerable_data) > 0:
                plt.figure(figsize=(10, 6))
                answerable_types = list(answerable_data.keys())
                success_rates = list(answerable_data.values())
                
                # Convert boolean to string for better display
                display_types = ['Non-Answerable' if not x else 'Answerable' for x in answerable_types]
                
                bars = plt.bar(display_types, success_rates, color=['lightcoral', 'lightgreen'], edgecolor='navy')
                plt.title('Success Rate by Answerability', fontsize=18, fontweight='bold', pad=30)
                plt.xlabel('Answerability', fontsize=14)
                plt.ylabel('Success Rate (%)', fontsize=14)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                
                # Add value labels on bars
                for bar, rate in zip(bars, success_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                            f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                # Set y-axis to percentage
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                plt.ylim(0, max(success_rates) * 1.2 if success_rates else 1)
                
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'success_by_answerable.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated success rate by answerable chart")
            else:
                logger.debug("No answerable data available for visualization")
        
        # Success rate by distractor_type
        if 'distractor_type' in metadata_analysis and metadata_analysis['distractor_type']:
            distractor_type_data = metadata_analysis['distractor_type'].get('success_rate', {})
            if distractor_type_data and len(distractor_type_data) > 0:
                plt.figure(figsize=(14, 8))
                distractor_types = list(distractor_type_data.keys())
                success_rates = list(distractor_type_data.values())
                
                # Sort by success rate for better visualization
                sorted_data = sorted(zip(distractor_types, success_rates), key=lambda x: x[1], reverse=True)
                distractor_types, success_rates = zip(*sorted_data)
                
                # Define colors for different distractor types
                color_map = {
                    'none': 'lightgreen',
                    'irrelevant': 'orange', 
                    'misleading': 'red',
                    'ambiguous': 'purple'
                }
                colors = [color_map.get(dt, 'lightblue') for dt in distractor_types]
                
                bars = plt.bar(range(len(distractor_types)), success_rates, color=colors, 
                              edgecolor='black', linewidth=1.5)
                plt.title('Success Rate by Distractor Type', fontsize=18, fontweight='bold', pad=30)
                plt.xlabel('Distractor Type', fontsize=14)
                plt.ylabel('Success Rate (%)', fontsize=14)
                plt.xticks(range(len(distractor_types)), distractor_types, rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                
                # Dynamic Y-axis range for better differentiation
                max_rate = max(success_rates) if success_rates else 0
                y_max = min(1.0, max_rate * 1.2)  # 20% above max rate, but not exceed 100%
                plt.ylim(0, y_max)
                # Format y-axis as percentage
                plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
                
                # Add value labels on bars with percentage
                for i, (bar, rate) in enumerate(zip(bars, success_rates)):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                            f'{rate:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(self.analysis_path / 'success_by_distractor_type.png', dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("Generated success rate by distractor_type chart")
            else:
                logger.debug("No distractor_type data available for visualization")
    
    def generate_analysis_report(self, data: List[Dict[str, Any]], stats: Dict[str, Any]):
        """Generate comprehensive analysis report."""
        report = {
            'analysis_metadata': {
                'analysis_name': self.analysis_name,
                'generated_at': datetime.now().isoformat(),
                'total_records': len(data),
                'data_sources': stats.get('data_sources', [])
            },
            'introduction': {
                'title': self.output_config.get('introduction', {}).get('title', 'Experiment Analysis'),
                'description': self.output_config.get('introduction', {}).get('description', ''),
                'context': self.output_config.get('introduction', {}).get('context', {})
            },
            'summary_statistics': stats,
            'detailed_analysis': {
                'success_rate_analysis': self._analyze_success_rates(data),
                'error_analysis': self._analyze_errors(data),
                'performance_analysis': self._analyze_performance(data)
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {str(k): convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        # Convert report to ensure JSON serialization
        converted_report = convert_numpy_types(report)
        
        # Save analysis report
        report_file = self.analysis_path / 'analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(converted_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis report saved to: {report_file}")
        return report
    
    def _analyze_success_rates(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze success rates by various dimensions."""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        analysis = {}
        
        if 'status' in df.columns:
            # Overall success rate
            total = len(df)
            success_count = len(df[df['status'] == 'SUCCESS'])
            analysis['overall_success_rate'] = success_count / total if total > 0 else 0
            
            # Success rate by question type (if available)
            if 'question_type' in df.columns:
                success_by_type = df.groupby('question_type')['status'].apply(
                    lambda x: (x == 'SUCCESS').sum() / len(x)
                ).to_dict()
                analysis['success_by_question_type'] = success_by_type
        
        return analysis
    
    def _analyze_errors(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error patterns and distributions."""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        analysis = {}
        
        if 'error_code' in df.columns:
            # Error code distribution
            error_dist = df['error_code'].value_counts().to_dict()
            analysis['error_code_distribution'] = error_dist
            
            # Most common errors
            analysis['most_common_errors'] = list(error_dist.keys())[:5]
        
        return analysis
    
    def _analyze_performance(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics."""
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        analysis = {}
        
        if 'execution_time' in df.columns:
            # Time statistics
            analysis['time_statistics'] = {
                'mean': df['execution_time'].mean(),
                'median': df['execution_time'].median(),
                'std': df['execution_time'].std(),
                'min': df['execution_time'].min(),
                'max': df['execution_time'].max()
            }
        
        return analysis
    
    def _analyze_by_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by metadata dimensions."""
        analysis = {}
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {list(df.columns)}")
        
        # Define metadata dimensions to analyze
        metadata_dimensions = [
            'reasoning_type', 'answerable', 'answer_type', 
            'complexity', 'distractor_type', 'has_distractor', 'table_number', 'domain'
        ]
        
        for dimension in metadata_dimensions:
            if dimension in df.columns:
                # Success rate by dimension based on LLM validation (only True counts as success)
                if 'llm_decision' in df.columns:
                    success_by_dim = df.groupby(dimension)['llm_decision'].apply(
                        lambda x: (x == True).sum() / len(x) if len(x) > 0 else 0
                    ).to_dict()
                else:
                    success_by_dim = {}
                
                # Count by dimension
                count_by_dim = df[dimension].value_counts().to_dict()
                
                # Error distribution by dimension
                error_by_dim = {}
                if 'error_code' in df.columns:
                    for value in df[dimension].unique():
                        subset = df[df[dimension] == value]
                        if len(subset) > 0:
                            error_by_dim[value] = subset['error_code'].value_counts().to_dict()
                
                # Time statistics by dimension
                time_by_dim = {}
                time_column = None
                if 'execution_time' in df.columns:
                    time_column = 'execution_time'
                elif 'time_s' in df.columns:
                    time_column = 'time_s'
                
                if time_column:
                    for value in df[dimension].unique():
                        subset = df[df[dimension] == value]
                        if len(subset) > 0:
                            time_by_dim[value] = {
                                'mean': subset[time_column].mean(),
                                'median': subset[time_column].median(),
                                'std': subset[time_column].std(),
                                'min': subset[time_column].min(),
                                'max': subset[time_column].max()
                            }
                
                analysis[dimension] = {
                    'success_rate': success_by_dim,
                    'count': count_by_dim,
                    'error_distribution': error_by_dim,
                    'time_statistics': time_by_dim
                }
        
        return analysis
    
    def _analyze_consistency(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency across multiple runs."""
        # Group by question_id to analyze multiple runs
        question_groups = {}
        for record in data:
            question_id = record.get('question_id')
            if question_id not in question_groups:
                question_groups[question_id] = []
            question_groups[question_id].append(record)
        
        consistency_stats = {
            'total_questions_with_multiple_runs': 0,
            'consistent_questions': 0,
            'inconsistent_questions': 0,
            'all_success_questions': 0,
            'all_error_questions': 0,
            'mixed_questions': 0,
            'consistency_rate': 0.0,
            'all_success_rate': 0.0,
            'all_error_rate': 0.0,
            'question_consistency_details': {}
        }
        
        for question_id, runs in question_groups.items():
            if len(runs) > 1:
                consistency_stats['total_questions_with_multiple_runs'] += 1
                
                # Check if all runs have the same LLM validation result
                llm_decisions = []
                for run in runs:
                    llm_valid = run.get('llm_valid', {})
                    if isinstance(llm_valid, dict):
                        decision = llm_valid.get('decision')
                        llm_decisions.append(decision)
                    else:
                        llm_decisions.append(None)
                
                is_consistent = len(set(llm_decisions)) == 1
                
                if is_consistent:
                    consistency_stats['consistent_questions'] += 1
                    # Check if all success or all error
                    if all(d == True for d in llm_decisions):
                        consistency_stats['all_success_questions'] += 1
                    elif all(d == False for d in llm_decisions):
                        consistency_stats['all_error_questions'] += 1
                else:
                    consistency_stats['inconsistent_questions'] += 1
                    consistency_stats['mixed_questions'] += 1
                
                # Store detailed consistency info
                consistency_stats['question_consistency_details'][question_id] = {
                    'total_runs': len(runs),
                    'llm_decisions': llm_decisions,
                    'is_consistent': is_consistent,
                    'consistency_type': 'all_success' if all(d == True for d in llm_decisions) else 
                                      'all_error' if all(d == False for d in llm_decisions) else 'mixed',
                    'run_details': [
                        {
                            'run_index': run.get('run_index', i),
                            'llm_decision': run.get('llm_valid', {}).get('decision') if isinstance(run.get('llm_valid'), dict) else None,
                            'status': run.get('status'),
                            'error_code': run.get('error_code')
                        }
                        for i, run in enumerate(runs)
                    ]
                }
        
        # Calculate rates
        if consistency_stats['total_questions_with_multiple_runs'] > 0:
            consistency_stats['consistency_rate'] = (
                consistency_stats['consistent_questions'] / 
                consistency_stats['total_questions_with_multiple_runs']
            )
            consistency_stats['all_success_rate'] = (
                consistency_stats['all_success_questions'] / 
                consistency_stats['total_questions_with_multiple_runs']
            )
            consistency_stats['all_error_rate'] = (
                consistency_stats['all_error_questions'] / 
                consistency_stats['total_questions_with_multiple_runs']
            )
        
        return consistency_stats
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting experiment analysis...")
        
        # Load data
        data = self.load_experiment_data()
        if not data:
            logger.error("No experiment data found")
            return
        
        # Generate statistics
        stats = self.generate_summary_statistics(data)
        
        # Log key statistics in a clean format
        logger.info(f"Analysis Summary:")
        logger.info(f"  Total questions: {stats.get('total_questions', 0)}")
        logger.info(f"  Unique questions: {stats.get('unique_questions', 0)}")
        logger.info(f"  Success rate: {stats.get('success_rate', 0):.1%}")
        
        result_dist = stats.get('result_distribution', {})
        if result_dist:
            logger.info(f"  Result distribution: {dict(result_dist)}")
        
        dsl_errors = stats.get('dsl_error_distribution', {})
        if dsl_errors:
            logger.info(f"  DSL execution errors: {dict(dsl_errors)}")
        
        llm_round_stats = stats.get('llm_round_stats', {})
        if llm_round_stats:
            logger.info(f"  LLM rounds: {llm_round_stats}")
        
        time_stats = stats.get('time_statistics', {})
        if time_stats:
            logger.info(f"  Time statistics: {time_stats}")
        
        consistency = stats.get('consistency_analysis', {})
        if consistency:
            logger.info(f"  Consistency rate: {consistency.get('consistency_rate', 0):.1%}")
        
        # Generate visualizations
        if self.statistics_config.get('basic_stats', True):
            self.generate_visualizations(data, stats)
        
        # Generate analysis report
        report = self.generate_analysis_report(data, stats)
        
        logger.info(f"Analysis complete. Results saved to: {self.analysis_path}")
        return report


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/analysis.yaml",
        help="Path to analysis configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        help="Override analysis name from config"
    )
    
    args = parser.parse_args()
    
    # Load configuration (skip validation for analysis config)
    try:
        config = load_config(args.config, validate_schema=False)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.output_dir:
        config['analysis']['output']['output_dir'] = args.output_dir
    if args.analysis_name:
        config['analysis']['output']['analysis_name'] = args.analysis_name
    
    # Run analysis
    try:
        analyzer = ExperimentAnalyzer(config)
        analyzer.run_analysis()
        print(f"Analysis complete. Results saved to: {analyzer.analysis_path}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
