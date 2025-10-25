#!/usr/bin/env python3
"""
Table Statistics Analysis Script
Analyzes and visualizes table data characteristics based on meta.json and quality_report.json files.
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

class TableStatisticsAnalyzer:
    def __init__(self, data_dir="data/tables", output_dir="experiments/analysis/table_statistics"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        # Remove existing directory and create fresh one
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load table index
        self.table_index = self._load_table_index()
        self.tables_data = []
        
    def _load_table_index(self):
        """Load table index for reference."""
        index_file = self.data_dir / "table_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}
    
    def load_table_data(self):
        """Load all table metadata and quality reports."""
        print("Loading table data...")
        
        for table_dir in self.data_dir.glob("table_*"):
            if table_dir.is_dir():
                meta_file = table_dir / "meta.json"
                quality_file = table_dir / "quality_report.json"
                
                if meta_file.exists() and quality_file.exists():
                    try:
                        with open(meta_file, 'r') as f:
                            meta_data = json.load(f)
                        with open(quality_file, 'r') as f:
                            quality_data = json.load(f)
                        
                        # Combine data
                        table_data = {
                            'table_id': meta_data.get('table_id'),
                            'title': meta_data.get('title'),
                            'description': meta_data.get('description'),
                            'columns': meta_data.get('columns', []),
                            'row_count': meta_data.get('row_count', 0),
                            'column_count': meta_data.get('column_count', 0),
                            'numeric_columns': meta_data.get('numeric_columns', []),
                            'has_time_series': meta_data.get('has_time_series', False),
                            'domain': meta_data.get('domain', 'unknown'),
                            'complexity_level': meta_data.get('complexity_level', 'unknown'),
                            'suitable_for_reasoning': meta_data.get('suitable_for_reasoning', True),
                            
                            # Quality report data
                            'original_shape': quality_data.get('original_shape', [0, 0]),
                            'final_shape': quality_data.get('final_shape', [0, 0]),
                            'rows_removed': quality_data.get('rows_removed', 0),
                            'columns_removed': quality_data.get('columns_removed', 0),
                            'data_completeness': quality_data.get('data_completeness', 0.0),
                            'numeric_columns_count': quality_data.get('numeric_columns', 0),
                            'success': quality_data.get('success', False)
                        }
                        
                        # Calculate derived metrics
                        table_data['numeric_ratio'] = table_data['numeric_columns_count'] / table_data['column_count'] if table_data['column_count'] > 0 else 0
                        table_data['size_category'] = self._categorize_size(table_data['row_count'])
                        table_data['original_rows'] = table_data['original_shape'][0] if table_data['original_shape'] else 0
                        table_data['original_columns'] = table_data['original_shape'][1] if table_data['original_shape'] else 0
                        
                        self.tables_data.append(table_data)
                        
                    except Exception as e:
                        print(f"Error loading {table_dir}: {e}")
        
        print(f"Loaded {len(self.tables_data)} tables")
        return pd.DataFrame(self.tables_data)
    
    def _categorize_size(self, row_count):
        """Categorize table size."""
        if row_count < 100:
            return 'small'
        elif row_count <= 500:
            return 'medium'
        else:
            return 'large'
    
    def analyze_structure(self, df):
        """Analyze table structure characteristics."""
        print("Analyzing table structure...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Table Structure Analysis', fontsize=16, fontweight='bold')
        
        # 1. Column count distribution
        axes[0].hist(df['column_count'], bins=20, alpha=0.7, edgecolor='black', color='#ff9999')
        axes[0].set_title('Column Count Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Columns')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(df['column_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["column_count"].mean():.1f}')
        axes[0].legend()
        
        # 2. Row count distribution
        axes[1].hist(df['row_count'], bins=20, alpha=0.7, edgecolor='black', color='#66b3ff')
        axes[1].set_title('Row Count Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Rows')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(df['row_count'].mean(), color='red', linestyle='--', label=f'Mean: {df["row_count"].mean():.1f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'table_structure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print("\n=== Table Structure Statistics ===")
        print(f"Total tables: {len(df)}")
        print(f"Average columns: {df['column_count'].mean():.1f} ± {df['column_count'].std():.1f}")
        print(f"Average rows: {df['row_count'].mean():.1f} ± {df['row_count'].std():.1f}")
        print(f"Average numeric ratio: {df['numeric_ratio'].mean():.2f} ± {df['numeric_ratio'].std():.2f}")
        print(f"Average data completeness: {df['data_completeness'].mean():.2f} ± {df['data_completeness'].std():.2f}")
        print(f"Time series tables: {df['has_time_series'].sum()}/{len(df)} ({df['has_time_series'].mean()*100:.1f}%)")
    
    def analyze_content(self, df):
        """Analyze table content characteristics."""
        print("Analyzing table content...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Table Content Analysis', fontsize=16, fontweight='bold')
        
        # 1. Domain distribution
        domain_counts = df['domain'].value_counts()
        colors = plt.cm.Set3(range(len(domain_counts)))
        axes[0].pie(domain_counts.values, labels=domain_counts.index, autopct='%1.1f%%', colors=colors)
        axes[0].set_title('Domain Distribution')
        
        # 2. Complexity level distribution (ordered: low, medium, high)
        complexity_counts = df['complexity_level'].value_counts()
        # Reorder to low, medium, high
        ordered_levels = ['low', 'medium', 'high']
        ordered_counts = [complexity_counts.get(level, 0) for level in ordered_levels]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        bars = axes[1].bar(ordered_levels, ordered_counts, color=colors)
        axes[1].set_title('Complexity Level Distribution')
        axes[1].set_xlabel('Complexity Level')
        axes[1].set_ylabel('Count')
        
        # Add count labels on bars
        for bar, count in zip(bars, ordered_counts):
            if count > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                           str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'table_content_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        print("\n=== Table Content Statistics ===")
        print("Domain distribution:")
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} ({count/len(df)*100:.1f}%)")
        
        print("\nComplexity level distribution:")
        for level, count in complexity_counts.items():
            print(f"  {level}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\nData cleaning:")
        print(f"  Average rows removed: {df['rows_removed'].mean():.1f} ± {df['rows_removed'].std():.1f}")
        print(f"  Average columns removed: {df['columns_removed'].mean():.1f} ± {df['columns_removed'].std():.1f}")
        print(f"  Average numeric columns: {df['numeric_columns_count'].mean():.1f} ± {df['numeric_columns_count'].std():.1f}")
    
    def analyze_relationships(self, df):
        """Analyze relationships between table characteristics."""
        print("Analyzing table relationships...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Numeric Ratio by Domain', fontsize=16, fontweight='bold')
        
        # Numeric ratio vs Domain
        sns.boxplot(data=df, x='domain', y='numeric_ratio', ax=ax)
        ax.set_title('Numeric Column Ratio by Domain', fontsize=14, fontweight='bold')
        ax.set_xlabel('Domain')
        ax.set_ylabel('Numeric Column Ratio')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'table_relationships_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print correlation analysis
        print("\n=== Table Relationships Statistics ===")
        print("Correlation between row count and data completeness:")
        print(f"  Pearson correlation: {df['row_count'].corr(df['data_completeness']):.3f}")
        print("Correlation between column count and numeric ratio:")
        print(f"  Pearson correlation: {df['column_count'].corr(df['numeric_ratio']):.3f}")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report."""
        print("Generating summary report...")
        
        report = {
            "summary": {
                "total_tables": len(df),
                "average_rows": float(df['row_count'].mean()),
                "average_columns": float(df['column_count'].mean()),
                "average_data_completeness": float(df['data_completeness'].mean()),
                "time_series_tables": int(df['has_time_series'].sum()),
                "time_series_percentage": float(df['has_time_series'].mean() * 100)
            },
            "domain_distribution": df['domain'].value_counts().to_dict(),
            "complexity_distribution": df['complexity_level'].value_counts().to_dict(),
            "size_distribution": df['size_category'].value_counts().to_dict(),
            "data_quality": {
                "average_rows_removed": float(df['rows_removed'].mean()),
                "average_columns_removed": float(df['columns_removed'].mean()),
                "average_numeric_columns": float(df['numeric_columns_count'].mean()),
                "average_numeric_ratio": float(df['numeric_ratio'].mean())
            },
            "calculation_standards": {
                "table_size_categories": {
                    "small": "row_count < 100",
                    "medium": "100 <= row_count <= 500", 
                    "large": "row_count > 500"
                },
                "complexity_levels": {
                    "low": "complexity_level = 'low'",
                    "medium": "complexity_level = 'medium'",
                    "high": "complexity_level = 'high'"
                }
            }
        }
        
        # Save report
        with open(self.output_dir / 'table_statistics_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n=== Summary Report ===")
        print(f"Total tables: {report['summary']['total_tables']}")
        print(f"Average rows: {report['summary']['average_rows']:.1f}")
        print(f"Average columns: {report['summary']['average_columns']:.1f}")
        print(f"Average data completeness: {report['summary']['average_data_completeness']:.2f}")
        print(f"Time series tables: {report['summary']['time_series_tables']} ({report['summary']['time_series_percentage']:.1f}%)")
        
        return report
    
    def run_analysis(self):
        """Run complete table statistics analysis."""
        print("Starting table statistics analysis...")
        
        # Load data
        df = self.load_table_data()
        
        if df.empty:
            print("No table data found!")
            return
        
        # Run analyses
        self.analyze_structure(df)
        self.analyze_content(df)
        self.analyze_relationships(df)
        report = self.generate_summary_report(df)
        
        print(f"\nAnalysis complete! Results saved to: {self.output_dir}")
        return report

if __name__ == "__main__":
    analyzer = TableStatisticsAnalyzer()
    analyzer.run_analysis()