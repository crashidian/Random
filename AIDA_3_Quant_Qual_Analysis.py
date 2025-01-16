import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Define color schemes
physical_colors = ['#000000', '#1A3049', '#5C738B', '#A3B6CD', '#CAD6E5']
cyber_colors = ['#C4BFC0', '#9D9795', '#CFB991', '#DAAA00', '#DDB945']

# Define specific binary color schemes
cyber_binary_colors = {'LOW': '#A3B6CD', 'HIGH': '#1A3049'}
fleet_binary_colors = {'LOW': '#CFB991', 'HIGH': '#8B5E3C'}

# File paths
file_path = r'C:\Users\me\OneDrive\Documents\RCODI\AIDA3\UseCaseSurvey.csv'
output_dir = r'C:\Users\me\OneDrive\Documents\RCODI\AIDA3'

def create_vertical_bar_plot(data, counts, title, filename, colors):
    """Create vertical bar plot with improved typography and proper labels"""
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(range(len(counts)), counts.values, color=colors[:len(counts)])
    
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right')
    
    # Add proper axis labels
    ax.set_ylabel('Number of Selections', fontsize=12)
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_title(title, pad=20, fontsize=14)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    ax.grid(True, axis='y', alpha=0.3)
    plt.subplots_adjust(bottom=0.2, left=0.1, right=0.9)
    
    # Add total count in the top right
    total_responses = sum(counts.values)
    plt.text(0.95, 0.95, f'Total Items Selected: {total_responses}',
             transform=ax.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_binary_distribution_plot(data, columns, title, filename, color_scheme):
    """Create bar plot showing distribution of LOW/HIGH responses as percentages"""
    # Process the data
    processed_data = []
    for col in columns:
        if col in data.columns:
            values = data[col].str.upper().value_counts()
            total = values.sum()
            percentages = (values / total * 100).round(1)
            processed_data.append({
                'category': col.split('[')[-1].split(']')[0] if '[' in col else col,
                'LOW': percentages.get('LOW', 0),
                'HIGH': percentages.get('HIGH', 0)
            })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the bars
    categories = [item['category'] for item in processed_data]
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars with specified colors
    low_bars = ax.bar(x - width/2, [item['LOW'] for item in processed_data], 
                      width, label='LOW', color=color_scheme['LOW'])
    high_bars = ax.bar(x + width/2, [item['HIGH'] for item in processed_data], 
                       width, label='HIGH', color=color_scheme['HIGH'])
    
    # Customize the plot
    ax.set_ylabel('Percentage of Responses (%)', fontsize=12)
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add percentage labels on the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    
    add_labels(low_bars)
    add_labels(high_bars)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    
    # Adjust layout
    plt.subplots_adjust(bottom=0.2)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_plot(data, columns, title, filename, colors=None):
    """Create radar plot with mean, min, max values and proper labels"""
    if colors is None:
        colors = physical_colors
        
    valid_cols = [col for col in columns if col in data.columns]
    if not valid_cols:
        print(f"Warning: No valid columns found for {title}")
        return
    
    data_clean = pd.to_numeric(data[valid_cols].stack(), errors='coerce').unstack()
    
    means = data_clean.mean()
    mins = data_clean.min()
    maxs = data_clean.max()
    
    angles = np.linspace(0, 2*np.pi, len(valid_cols), endpoint=False)
    
    values_mean = means.values.tolist()
    values_min = mins.values.tolist()
    values_max = maxs.values.tolist()
    values_mean.append(values_mean[0])
    values_min.append(values_min[0])
    values_max.append(values_max[0])
    angles = np.concatenate((angles, [angles[0]]))
    
    # Create figure with larger size to accommodate labels
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data with specified colors and improved labels
    ax.plot(angles, values_mean, 'o-', linewidth=2, label='Mean Score', color=colors[0])
    ax.fill(angles, values_mean, alpha=0.25, color=colors[0])
    
    ax.plot(angles, values_min, 'o-', linewidth=2, label='Minimum Score', color=colors[1])
    ax.fill(angles, values_min, alpha=0.25, color=colors[1])
    
    ax.plot(angles, values_max, 'o-', linewidth=2, label='Maximum Score', color=colors[2])
    ax.fill(angles, values_max, alpha=0.25, color=colors[2])
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Improved axis labels
    ax.set_xticks(angles[:-1])
    labels = [col.split('(')[0].strip() if '(' in col else col for col in valid_cols]
    labels = [label[:30] + '...' if len(label) > 30 else label for label in labels]
    ax.set_xticklabels(labels, fontsize=8)
    
    # Add mean value labels with improved positioning
    for i, (angle, value, label) in enumerate(zip(angles[:-1], values_mean[:-1], labels)):
        # Calculate optimal label distance and position
        base_distance = 0.6
        sector = int((angle + np.pi/2) / (np.pi/4))
        angle_offset = 0
        
        if sector in [0, 7]:  # Top
            label_distance = value + base_distance
            va = 'bottom'
        elif sector in [1, 2]:  # Right
            label_distance = value + base_distance * 0.8
            angle_offset = -0.1
            va = 'center'
        elif sector in [3, 4]:  # Bottom
            label_distance = value + base_distance
            va = 'top'
        else:  # Left
            label_distance = value + base_distance * 0.8
            angle_offset = 0.1
            va = 'center'
        
        # Add the label with adjusted position
        ax.text(angle + angle_offset, label_distance, f'{value:.1f}',
                ha='center', va=va, fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    
    ax.set_ylim(0, 6)
    ax.set_yticks(np.arange(1, 6))
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=11)
    plt.title(f'{title}\n(Likert Scale of 1-5 with 1 being lowest ranking and 5 being highest ranking)', 
              y=1.05, pad=20, fontsize=14)
    
    # Add legend
    legend = plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    legend.set_title('Metrics')
    
    plt.subplots_adjust(right=0.8)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    try:
        # Read data
        df = pd.read_csv(file_path)
        print("Successfully loaded data")
        
        # 1. Problem Areas Distribution
        col_e = 'Problem areas: Please specify the problem area related to AAVs that your use case addresses. Please select the most applicable one! '
        column_e_counts = df[col_e].value_counts()
        create_vertical_bar_plot(df, column_e_counts, 'Distribution of AAV Problem Areas', 'problem_areas.png', physical_colors)
        print("Created problem areas plot")
        
        # 2. Technical Areas Distribution
        technical_cols = df.iloc[:, 5:12]
        tech_counts = pd.Series(technical_cols.notna().sum(), name='Count')
        create_vertical_bar_plot(df, tech_counts, 'Distribution of Technical Areas in AAV Use Cases', 'technical_areas.png', cyber_colors)
        print("Created technical areas plot")
        
        # 3. ML Areas Distribution
        ml_cols = df.iloc[:, 12:27]
        ml_counts = pd.Series(ml_cols.notna().sum(), name='Count')
        create_vertical_bar_plot(df, ml_counts, 'Distribution of Machine Learning Areas in AAV Use Cases', 'ml_areas.png', cyber_colors)
        print("Created ML areas plot")
        
        # 4. Physical Components Analysis (Likert Scale)
        physical_cols = df.columns[[31, 33, 35, 37, 44]]
        create_radar_plot(df, physical_cols, 'Physical Components Analysis\nEvaluation of Importance of Each Physical Component', 'physical_radar.png', colors=physical_colors)
        print("Created physical components radar plot")
        
        # 5. Cyber Components Analysis (Binary Scale - Bar Chart with Updated Colors)
        cyber_cols = df.columns[46:52]
        create_binary_distribution_plot(df, cyber_cols, 'Cyber Components Analysis\nDistribution of Importance Ratings', 
                                      'cyber_distribution.png', cyber_binary_colors)
        print("Created cyber components distribution plot")
        
        # 6. Fleet Vehicle Types Analysis (Binary Scale - Bar Chart with Updated Colors)
        fleet_cols = df.columns[[39, 40, 41, 42]]  # Columns AN, AO, AP, AQ
        create_binary_distribution_plot(df, fleet_cols, 'Fleet Vehicle Types Analysis\nDistribution of Importance Ratings', 
                                      'fleet_distribution.png', fleet_binary_colors)
        print("Created fleet vehicle types distribution plot")
        
        print("\nAnalysis complete. All files saved to output directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
