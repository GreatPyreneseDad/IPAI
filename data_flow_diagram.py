#!/usr/bin/env python3
"""
IPAI Data Flow Visualization
Generates a comprehensive data flow diagram for the IPAI system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np

def create_data_flow_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define colors
    input_color = '#E8F4FD'
    processing_color = '#B8E0D2'
    storage_color = '#D6CFC7'
    analytics_color = '#FADBD8'
    optimization_color = '#F9E79F'
    
    # Title
    ax.text(50, 95, 'IPAI Data Processing Flow', fontsize=20, fontweight='bold', ha='center')
    
    # Input Layer (Top)
    input_box = FancyBboxPatch((5, 80), 20, 10, boxstyle="round,pad=0.1", 
                               facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(15, 85, 'User Messages\n& API Input', ha='center', va='center', fontsize=10)
    
    # Message Analysis
    analysis_box = FancyBboxPatch((35, 80), 20, 10, boxstyle="round,pad=0.1",
                                  facecolor=processing_color, edgecolor='black', linewidth=2)
    ax.add_patch(analysis_box)
    ax.text(45, 85, 'Message\nCoherence\nAnalyzer', ha='center', va='center', fontsize=10)
    
    # Text Statistics
    stats_box = FancyBboxPatch((35, 65), 20, 8, boxstyle="round,pad=0.1",
                               facecolor=processing_color, edgecolor='gray', linewidth=1)
    ax.add_patch(stats_box)
    ax.text(45, 69, 'Text Statistics:\n• Readability\n• Patterns\n• Red Flags', 
            ha='center', va='center', fontsize=8)
    
    # GCT Calculator
    gct_box = FancyBboxPatch((65, 70), 25, 15, boxstyle="round,pad=0.1",
                             facecolor=processing_color, edgecolor='black', linewidth=2)
    ax.add_patch(gct_box)
    ax.text(77.5, 77.5, 'GCT Calculator\n\nψ, ρ, q, f → C\n\nOptimization:\nk_m, k_i params', 
            ha='center', va='center', fontsize=10)
    
    # Cache Layer
    cache_box = FancyBboxPatch((65, 55), 25, 8, boxstyle="round,pad=0.1",
                               facecolor=optimization_color, edgecolor='orange', linewidth=2)
    ax.add_patch(cache_box)
    ax.text(77.5, 59, 'LRU Cache\n(TTL Eviction)', ha='center', va='center', fontsize=9)
    
    # Triadic Processor
    triadic_box = FancyBboxPatch((35, 45), 25, 12, boxstyle="round,pad=0.1",
                                 facecolor=processing_color, edgecolor='black', linewidth=2)
    ax.add_patch(triadic_box)
    ax.text(47.5, 51, 'Triadic Processor\n\n1. Generate\n2. Analyze\n3. Ground', 
            ha='center', va='center', fontsize=10)
    
    # Database
    db_box = FancyBboxPatch((10, 25), 30, 12, boxstyle="round,pad=0.1",
                            facecolor=storage_color, edgecolor='black', linewidth=2)
    ax.add_patch(db_box)
    ax.text(25, 31, 'PostgreSQL Database\n\n• Coherence Profiles\n• User Data\n• Analytics', 
            ha='center', va='center', fontsize=10)
    
    # Performance Monitor
    perf_box = FancyBboxPatch((70, 35), 20, 10, boxstyle="round,pad=0.1",
                              facecolor=optimization_color, edgecolor='orange', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(80, 40, 'Performance\nMonitor\n\n• Metrics\n• Slow Queries', 
            ha='center', va='center', fontsize=9)
    
    # Analytics Engine
    analytics_box = FancyBboxPatch((45, 25), 20, 12, boxstyle="round,pad=0.1",
                                   facecolor=analytics_color, edgecolor='black', linewidth=2)
    ax.add_patch(analytics_box)
    ax.text(55, 31, 'Analytics\nEngine\n\n• Trends\n• Predictions\n• Insights', 
            ha='center', va='center', fontsize=10)
    
    # API Output
    output_box = FancyBboxPatch((75, 10), 20, 10, boxstyle="round,pad=0.1",
                                facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(85, 15, 'API Response\n& Dashboards', ha='center', va='center', fontsize=10)
    
    # Add arrows showing data flow
    arrows = [
        # Input to Analysis
        ((25, 85), (35, 85)),
        # Analysis to Stats
        ((45, 80), (45, 73)),
        # Analysis to GCT
        ((55, 85), (65, 80)),
        # Stats to GCT
        ((55, 69), (65, 73)),
        # GCT to Cache
        ((77.5, 70), (77.5, 63)),
        # Cache feedback to GCT
        ((77.5, 55), (77.5, 70)),
        # GCT to Triadic
        ((65, 75), (60, 55)),
        # Triadic to DB
        ((35, 48), (25, 37)),
        # DB to Analytics
        ((40, 31), (45, 31)),
        # Analytics to Output
        ((65, 31), (75, 20)),
        # GCT to Performance
        ((85, 70), (85, 45)),
        # Performance to DB
        ((70, 37), (40, 32)),
        # Triadic feedback loop
        ((47.5, 45), (47.5, 57)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, connectionstyle="arc3,rad=0.2",
                                arrowstyle='->', mutation_scale=20, linewidth=2,
                                color='#2C3E50', alpha=0.7)
        ax.add_patch(arrow)
    
    # Add performance metrics annotations
    ax.text(5, 15, 'Key Performance Metrics:', fontsize=12, fontweight='bold')
    ax.text(5, 12, '• Cache Hit Rate: Track in CacheManager', fontsize=9)
    ax.text(5, 10, '• Calculation Time: < 100ms target', fontsize=9)
    ax.text(5, 8, '• Query Performance: Monitored via slow_queries', fontsize=9)
    ax.text(5, 6, '• Memory Usage: GC triggered at 80%', fontsize=9)
    
    # Add optimization notes
    ax.text(5, 2, 'Optimization Opportunities:', fontsize=12, fontweight='bold', color='red')
    ax.text(5, 0, '1. Vectorize GCT calculations  2. Implement Redis for real-time  3. Add ML predictions', 
            fontsize=9, color='red')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=input_color, edgecolor='black', label='Input/Output'),
        patches.Patch(facecolor=processing_color, edgecolor='black', label='Processing'),
        patches.Patch(facecolor=storage_color, edgecolor='black', label='Storage'),
        patches.Patch(facecolor=analytics_color, edgecolor='black', label='Analytics'),
        patches.Patch(facecolor=optimization_color, edgecolor='orange', label='Optimization')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.title('IPAI System Data Flow and Processing Pipeline', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('/Users/chris/IPAI/ipai_data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Users/chris/IPAI/ipai_data_flow_diagram.pdf', bbox_inches='tight')
    print("Data flow diagram saved as ipai_data_flow_diagram.png and .pdf")

def create_performance_bottleneck_diagram():
    """Create a diagram showing performance bottlenecks and solutions"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, 'IPAI Performance Bottlenecks & Solutions', fontsize=18, fontweight='bold', ha='center')
    
    # Define colors
    bottleneck_color = '#FFB6C1'
    solution_color = '#90EE90'
    
    # Bottleneck 1: Text Analysis
    bottleneck1 = FancyBboxPatch((5, 75), 40, 15, boxstyle="round,pad=0.1",
                                 facecolor=bottleneck_color, edgecolor='red', linewidth=2)
    ax.add_patch(bottleneck1)
    ax.text(25, 85, 'Bottleneck: Text Analysis', fontsize=12, fontweight='bold', ha='center')
    ax.text(25, 81, '• Sequential regex processing', fontsize=9, ha='center')
    ax.text(25, 79, '• CPU-intensive pattern matching', fontsize=9, ha='center')
    ax.text(25, 77, '• No parallelization', fontsize=9, ha='center')
    
    solution1 = FancyBboxPatch((55, 75), 40, 15, boxstyle="round,pad=0.1",
                               facecolor=solution_color, edgecolor='green', linewidth=2)
    ax.add_patch(solution1)
    ax.text(75, 85, 'Solution', fontsize=12, fontweight='bold', ha='center')
    ax.text(75, 81, '• ThreadPoolExecutor for parallel processing', fontsize=9, ha='center')
    ax.text(75, 79, '• Compiled regex patterns', fontsize=9, ha='center')
    ax.text(75, 77, '• Batch message processing', fontsize=9, ha='center')
    
    # Bottleneck 2: GCT Calculations
    bottleneck2 = FancyBboxPatch((5, 55), 40, 15, boxstyle="round,pad=0.1",
                                 facecolor=bottleneck_color, edgecolor='red', linewidth=2)
    ax.add_patch(bottleneck2)
    ax.text(25, 65, 'Bottleneck: GCT Calculations', fontsize=12, fontweight='bold', ha='center')
    ax.text(25, 61, '• Complex non-linear operations', fontsize=9, ha='center')
    ax.text(25, 59, '• Per-profile computation', fontsize=9, ha='center')
    ax.text(25, 57, '• Cache misses on unique params', fontsize=9, ha='center')
    
    solution2 = FancyBboxPatch((55, 55), 40, 15, boxstyle="round,pad=0.1",
                               facecolor=solution_color, edgecolor='green', linewidth=2)
    ax.add_patch(solution2)
    ax.text(75, 65, 'Solution', fontsize=12, fontweight='bold', ha='center')
    ax.text(75, 61, '• NumPy vectorization', fontsize=9, ha='center')
    ax.text(75, 59, '• GPU acceleration (CUDA)', fontsize=9, ha='center')
    ax.text(75, 57, '• Expanded cache size & strategy', fontsize=9, ha='center')
    
    # Bottleneck 3: Database Operations
    bottleneck3 = FancyBboxPatch((5, 35), 40, 15, boxstyle="round,pad=0.1",
                                 facecolor=bottleneck_color, edgecolor='red', linewidth=2)
    ax.add_patch(bottleneck3)
    ax.text(25, 45, 'Bottleneck: Database Queries', fontsize=12, fontweight='bold', ha='center')
    ax.text(25, 41, '• N+1 query patterns', fontsize=9, ha='center')
    ax.text(25, 39, '• Multiple aggregation queries', fontsize=9, ha='center')
    ax.text(25, 37, '• No query result caching', fontsize=9, ha='center')
    
    solution3 = FancyBboxPatch((55, 35), 40, 15, boxstyle="round,pad=0.1",
                               facecolor=solution_color, edgecolor='green', linewidth=2)
    ax.add_patch(solution3)
    ax.text(75, 45, 'Solution', fontsize=12, fontweight='bold', ha='center')
    ax.text(75, 41, '• Materialized views', fontsize=9, ha='center')
    ax.text(75, 39, '• Query optimization with CTEs', fontsize=9, ha='center')
    ax.text(75, 37, '• Redis for real-time metrics', fontsize=9, ha='center')
    
    # Bottleneck 4: Memory Management
    bottleneck4 = FancyBboxPatch((5, 15), 40, 15, boxstyle="round,pad=0.1",
                                 facecolor=bottleneck_color, edgecolor='red', linewidth=2)
    ax.add_patch(bottleneck4)
    ax.text(25, 25, 'Bottleneck: Memory Usage', fontsize=12, fontweight='bold', ha='center')
    ax.text(25, 21, '• Large trajectory storage', fontsize=9, ha='center')
    ax.text(25, 19, '• Unbounded cache growth', fontsize=9, ha='center')
    ax.text(25, 17, '• Full history processing', fontsize=9, ha='center')
    
    solution4 = FancyBboxPatch((55, 15), 40, 15, boxstyle="round,pad=0.1",
                               facecolor=solution_color, edgecolor='green', linewidth=2)
    ax.add_patch(solution4)
    ax.text(75, 25, 'Solution', fontsize=12, fontweight='bold', ha='center')
    ax.text(75, 21, '• Sliding window analysis', fontsize=9, ha='center')
    ax.text(75, 19, '• LRU cache with size limits', fontsize=9, ha='center')
    ax.text(75, 17, '• Stream processing architecture', fontsize=9, ha='center')
    
    # Add arrows
    for y in [82.5, 62.5, 42.5, 22.5]:
        arrow = FancyArrowPatch((45, y), (55, y), arrowstyle='->', mutation_scale=20,
                                linewidth=2, color='blue')
        ax.add_patch(arrow)
    
    # Add performance impact
    ax.text(50, 8, 'Expected Performance Improvements:', fontsize=12, fontweight='bold', ha='center')
    ax.text(50, 5, '• 5-10x speedup in text analysis  • 3-5x faster GCT calculations', fontsize=10, ha='center')
    ax.text(50, 3, '• 50% reduction in database load  • 40% less memory usage', fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('/Users/chris/IPAI/ipai_bottlenecks_solutions.png', dpi=300, bbox_inches='tight')
    print("Bottlenecks diagram saved as ipai_bottlenecks_solutions.png")

if __name__ == "__main__":
    create_data_flow_diagram()
    create_performance_bottleneck_diagram()
    print("All diagrams created successfully!")