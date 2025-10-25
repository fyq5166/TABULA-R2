#!/usr/bin/env python3
"""
Question Statistics Analysis Script
Analyzes and visualizes question data characteristics based on question JSON files.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QuestionStatisticsAnalyzer:
    def __init__(self, data_dir="data/questions", output_dir="experiments/analysis/question_statistics"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        # Remove existing directory and create fresh one
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.questions_data = []
        
    def load_question_data(self):
        """Load all question data from single_table and multi_table directories."""
        print("Loading question data...")
        
        # Load single_table questions
        single_table_dir = self.data_dir / "single_table"
        if single_table_dir.exists():
            for question_file in single_table_dir.glob("question_*.json"):
                self._load_question_file(question_file, "single_table")
        
        # Load multi_table questions
        multi_table_dir = self.data_dir / "multi_table"
        if multi_table_dir.exists():
            for topic_dir in multi_table_dir.iterdir():
                if topic_dir.is_dir():
                    for question_file in topic_dir.glob("question_*.json"):
                        self._load_question_file(question_file, "multi_table")
        
        # Load distractor_bank questions
        distractor_dir = self.data_dir / "distractor_bank"
        if distractor_dir.exists():
            for question_dir in distractor_dir.iterdir():
                if question_dir.is_dir():
                    for question_file in question_dir.glob("question_*.json"):
                        self._load_question_file(question_file, "distractor_bank")
        
        print(f"Loaded {len(self.questions_data)} questions")
        return pd.DataFrame(self.questions_data)
    
    def _load_question_file(self, question_file, source_type):
        """Load a single question file."""
        try:
            with open(question_file, 'r') as f:
                question_data = json.load(f)
            
            # Extract basic information
            question_info = {
                'question_id': question_data.get('question_id'),
                'question': question_data.get('question', ''),
                'table_refs': question_data.get('table_refs', []),
                'answer': question_data.get('answer'),
                'answer_type': question_data.get('answer_type'),
                'domain': question_data.get('domain'),
                'reasoning_type': question_data.get('reasoning_type'),
                'answerable': question_data.get('answerable', True),
                'source_type': source_type,
                'question_file': str(question_file)
            }
            
            # Extract complexity metrics
            complexity_metrics = question_data.get('complexity_metrics', {})
            question_info.update({
                'rows_involved': complexity_metrics.get('rows_involved', 0),
                'columns_involved': complexity_metrics.get('columns_involved', 0),
                'steps_count': complexity_metrics.get('steps_count', 0),
                'complexity_score': complexity_metrics.get('complexity_score', 0.0)
            })
            
            # Extract additional fields
            question_info.update({
                'table_number': question_data.get('table_number', len(question_data.get('table_refs', []))),
                'requires_calculation': question_data.get('requires_calculation', False),
                'has_distractor': question_data.get('has_distractor', False),
                'distractor_type': question_data.get('distractor_type', 'none')
            })
            
            # Calculate derived metrics
            question_info['question_length'] = len(question_info['question'])
            question_info['reasoning_steps_count'] = len(question_data.get('reasoning_steps', []))
            question_info['complexity_level'] = self._categorize_complexity(question_info['complexity_score'])
            question_info['is_single_table'] = question_info['table_number'] == 1
            question_info['is_multi_table'] = question_info['table_number'] > 1
            
            self.questions_data.append(question_info)
            
        except Exception as e:
            print(f"Error loading {question_file}: {e}")
    
    def _categorize_complexity(self, complexity_score):
        """Categorize complexity level."""
        if complexity_score <= 6.0:
            return 'low'
        elif complexity_score <= 21.5:
            return 'medium'
        else:
            return 'high'
    
    def generate_individual_charts(self, df):
        """Generate individual charts as requested."""
        print("Generating individual charts...")
        
        # 1. Single vs Multi table pie chart
        self._create_single_vs_multi_chart(df)
        
        # 2. Reasoning type distribution bar chart
        self._create_reasoning_type_chart(df)
        
        # 3. Domain distribution pie chart
        self._create_domain_chart(df)
        
        # 4. Answer type distribution bar chart
        self._create_answer_type_chart(df)
        
        # 5. Source type distribution pie chart
        self._create_source_type_chart(df)
        
        # 6. Answerable questions pie chart
        self._create_answerable_chart(df)
        
        # 7. Reasoning steps distribution
        self._create_reasoning_steps_chart(df)
        
        # 8. Table number distribution (improved)
        self._create_table_number_chart(df)
        
        # 9. Domain vs Reasoning type heatmap
        self._create_domain_reasoning_heatmap(df)
        
        # 10. Complexity by answer type boxplot
        self._create_complexity_by_answer_type(df)
    
    def _create_single_vs_multi_chart(self, df):
        """Create single vs multi-table pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        table_type_counts = df['is_single_table'].value_counts()
        labels = ['Multi-table', 'Single-table']
        colors = ['#ff9999', '#66b3ff']
        
        wedges, texts, autotexts = ax.pie(table_type_counts.values, labels=labels, colors=colors, 
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(df))})')
        ax.set_title('Single vs Multi-table Questions', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'question_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_reasoning_type_chart(self, df):
        """Create reasoning type distribution bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        reasoning_counts = df['reasoning_type'].value_counts()
        bars = ax.bar(range(len(reasoning_counts)), reasoning_counts.values, color='skyblue', edgecolor='black')
        ax.set_title('Reasoning Type Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Reasoning Type')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(reasoning_counts)))
        ax.set_xticklabels(reasoning_counts.index, rotation=45)
        
        # Add count and percentage labels with more space
        max_height = max(reasoning_counts.values)
        ax.set_ylim(0, max_height * 1.15)  # Add 15% more space at top
        
        for i, (bar, count) in enumerate(zip(bars, reasoning_counts.values)):
            percentage = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_height * 0.02, 
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reasoning_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_domain_chart(self, df):
        """Create domain distribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        domain_counts = df['domain'].value_counts()
        colors = plt.cm.Set3(range(len(domain_counts)))
        
        wedges, texts, autotexts = ax.pie(domain_counts.values, labels=domain_counts.index, colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(df))})')
        ax.set_title('Domain Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_answer_type_chart(self, df):
        """Create answer type distribution bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        answer_type_counts = df['answer_type'].value_counts()
        bars = ax.bar(range(len(answer_type_counts)), answer_type_counts.values, color='lightgreen', edgecolor='black')
        ax.set_title('Answer Type Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Answer Type')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(answer_type_counts)))
        ax.set_xticklabels(answer_type_counts.index, rotation=45)
        
        # Add count and percentage labels with more space
        max_height = max(answer_type_counts.values)
        ax.set_ylim(0, max_height * 1.15)  # Add 15% more space at top
        
        for i, (bar, count) in enumerate(zip(bars, answer_type_counts.values)):
            percentage = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_height * 0.02, 
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'answer_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_source_type_chart(self, df):
        """Create source type distribution pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        source_counts = df['source_type'].value_counts()
        colors = ['#ffcc99', '#99ccff', '#99ff99']
        
        wedges, texts, autotexts = ax.pie(source_counts.values, labels=source_counts.index, colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(df))})')
        ax.set_title('Source Type Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'source_type_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_answerable_chart(self, df):
        """Create answerable questions pie chart."""
        fig, ax = plt.subplots(figsize=(8, 6))
        answerable_counts = df['answerable'].value_counts()
        labels = ['Not Answerable', 'Answerable']
        colors = ['#ff9999', '#99ff99']
        
        wedges, texts, autotexts = ax.pie(answerable_counts.values, labels=labels, colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(df))})')
        ax.set_title('Answerable Questions Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'answerable_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_reasoning_steps_chart(self, df):
        """Create reasoning steps distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        steps_counts = df['reasoning_steps_count'].value_counts().sort_index()
        bars = ax.bar(steps_counts.index, steps_counts.values, color='orange', edgecolor='black')
        ax.set_title('Reasoning Steps Count Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Reasoning Steps')
        ax.set_ylabel('Count')
        
        # Add count and percentage labels with more space
        max_height = max(steps_counts.values)
        ax.set_ylim(0, max_height * 1.15)  # Add 15% more space at top
        
        for bar, count in zip(bars, steps_counts.values):
            percentage = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_height * 0.02, 
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reasoning_steps_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_table_number_chart(self, df):
        """Create improved table number distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        table_counts = df['table_number'].value_counts().sort_index()
        bars = ax.bar(table_counts.index, table_counts.values, color='lightcoral', edgecolor='black', width=0.6)
        ax.set_title('Table Number Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Tables')
        ax.set_ylabel('Count')
        ax.set_xticks(table_counts.index)
        
        # Add count and percentage labels with more space
        max_height = max(table_counts.values)
        ax.set_ylim(0, max_height * 1.15)  # Add 15% more space at top
        
        for bar, count in zip(bars, table_counts.values):
            percentage = count / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_height * 0.02, 
                   f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'table_number_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_domain_reasoning_heatmap(self, df):
        """Create domain vs reasoning type heatmap."""
        fig, ax = plt.subplots(figsize=(10, 6))
        domain_reasoning = pd.crosstab(df['domain'], df['reasoning_type'])
        sns.heatmap(domain_reasoning, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Domain vs Reasoning Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Reasoning Type')
        ax.set_ylabel('Domain')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_vs_reasoning_type.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_complexity_by_answer_type(self, df):
        """Create complexity by answer type boxplot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='answer_type', y='complexity_score', ax=ax)
        ax.set_title('Complexity Score by Answer Type', fontsize=14, fontweight='bold')
        ax.set_xlabel('Answer Type')
        ax.set_ylabel('Complexity Score')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_by_answer_type.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_statistics(self, df):
        """Print comprehensive statistics."""
        print("\n=== Question Statistics ===")
        print(f"Total questions: {len(df)}")
        print(f"Single-table questions: {df['is_single_table'].sum()} ({df['is_single_table'].mean()*100:.1f}%)")
        print(f"Multi-table questions: {df['is_multi_table'].sum()} ({df['is_multi_table'].mean()*100:.1f}%)")
        print(f"Answerable questions: {df['answerable'].sum()} ({df['answerable'].mean()*100:.1f}%)")
        
        print("\nReasoning type distribution:")
        reasoning_counts = df['reasoning_type'].value_counts()
        for reasoning_type, count in reasoning_counts.items():
            print(f"  {reasoning_type}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nDomain distribution:")
        domain_counts = df['domain'].value_counts()
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nComplexity statistics:")
        print(f"Average complexity score: {df['complexity_score'].mean():.2f} ± {df['complexity_score'].std():.2f}")
        complexity_level_counts = df['complexity_level'].value_counts()
        for level, count in complexity_level_counts.items():
            print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\nContent statistics:")
        print(f"Average question length: {df['question_length'].mean():.1f} ± {df['question_length'].std():.1f}")
        print(f"Average reasoning steps: {df['reasoning_steps_count'].mean():.1f} ± {df['reasoning_steps_count'].std():.1f}")
        print(f"Average table number: {df['table_number'].mean():.1f} ± {df['table_number'].std():.1f}")
        print(f"Questions requiring calculation: {df['requires_calculation'].sum()} ({df['requires_calculation'].mean()*100:.1f}%)")
        
        print("\nCorrelation analysis:")
        print(f"Table number vs complexity: {df['table_number'].corr(df['complexity_score']):.3f}")
        print(f"Question length vs complexity: {df['question_length'].corr(df['complexity_score']):.3f}")
        print(f"Steps count vs complexity: {df['steps_count'].corr(df['complexity_score']):.3f}")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report."""
        print("Generating summary report...")
        
        report = {
            "summary": {
                "total_questions": len(df),
                "single_table_questions": int(df['is_single_table'].sum()),
                "multi_table_questions": int(df['is_multi_table'].sum()),
                "average_complexity": float(df['complexity_score'].mean()),
                "average_question_length": float(df['question_length'].mean()),
                "answerable_questions": int(df['answerable'].sum()),
                "questions_requiring_calculation": int(df['requires_calculation'].sum())
            },
            "type_distribution": {
                "reasoning_types": df['reasoning_type'].value_counts().to_dict(),
                "domains": df['domain'].value_counts().to_dict(),
                "answer_types": df['answer_type'].value_counts().to_dict(),
                "complexity_levels": df['complexity_level'].value_counts().to_dict()
            },
            "complexity_metrics": {
                "average_rows_involved": float(df['rows_involved'].mean()),
                "average_columns_involved": float(df['columns_involved'].mean()),
                "average_steps_count": float(df['steps_count'].mean()),
                "average_table_number": float(df['table_number'].mean())
            },
            "calculation_standards": {
                "complexity_levels": {
                    "low": "complexity_score <= 6.0",
                    "medium": "6.1 <= complexity_score <= 21.5",
                    "high": "complexity_score > 21.5"
                }
            }
        }
        
        # Add distractor analysis if available
        distractor_df = df[df['source_type'] == 'distractor_bank']
        if not distractor_df.empty:
            report["distractor_analysis"] = {
                "total_distractor_questions": len(distractor_df),
                "distractor_types": distractor_df['distractor_type'].value_counts().to_dict(),
                "average_complexity_distractor": float(distractor_df['complexity_score'].mean()),
                "average_table_number_distractor": float(distractor_df['table_number'].mean())
            }
        
        # Save report
        with open(self.output_dir / 'question_statistics_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Summary Report ===")
        print(f"Total questions: {report['summary']['total_questions']}")
        print(f"Single-table questions: {report['summary']['single_table_questions']} ({report['summary']['single_table_questions']/report['summary']['total_questions']*100:.1f}%)")
        print(f"Multi-table questions: {report['summary']['multi_table_questions']} ({report['summary']['multi_table_questions']/report['summary']['total_questions']*100:.1f}%)")
        print(f"Average complexity: {report['summary']['average_complexity']:.2f}")
        print(f"Answerable questions: {report['summary']['answerable_questions']} ({report['summary']['answerable_questions']/report['summary']['total_questions']*100:.1f}%)")
        
        return report
    
    def run_analysis(self):
        """Run complete question statistics analysis."""
        print("Starting question statistics analysis...")
        
        # Load data
        df = self.load_question_data()
        
        if df.empty:
            print("No question data found!")
            return
        
        # Generate individual charts
        self.generate_individual_charts(df)
        
        # Print statistics
        self.print_statistics(df)
        
        # Generate summary report
        report = self.generate_summary_report(df)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        return report

if __name__ == "__main__":
    analyzer = QuestionStatisticsAnalyzer()
    analyzer.run_analysis()