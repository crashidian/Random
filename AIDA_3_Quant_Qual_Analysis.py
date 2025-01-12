import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Define color schemes
physical_colors = ['#000000', '#1A3049', '#6C738B', '#A3B6CD', '#CAD6E5']
cyber_colors = ['#C4BFC0', '#9D9795', '#CFB991', '#DAAA00', '#DDB945']

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

def clean_and_convert_data(data, is_binary=False):
    """Clean and convert data to numeric values"""
    if is_binary:
        data = data.apply(lambda x: x.str.lower() if pd.api.types.is_string_dtype(x) else x)
        data = data.replace({
            'low': 1,
            'high': 2,
            'Low': 1,
            'High': 2,
            'LOW': 1,
            'HIGH': 2
        })
    return pd.to_numeric(data.stack(), errors='coerce').unstack()

def create_radar_plot(data, columns, title, filename, is_binary=False, colors=None):
    """Create radar plot with mean, min, max values and proper labels"""
    if colors is None:
        colors = physical_colors
        
    valid_cols = [col for col in columns if col in data.columns]
    if not valid_cols:
        print(f"Warning: No valid columns found for {title}")
        return
    
    data_clean = clean_and_convert_data(data[valid_cols], is_binary)
    
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
    
    fig = plt.figure(figsize=(12, 8))
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
    
    if is_binary:
        ax.set_ylim(0, 2)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Low', 'High'], fontsize=11)
        plt.title(f'{title}\n(Binary Scale of Low and High (Low = 1, High = 2)', y=1.05, pad=20, fontsize=14)
    else:
        ax.set_ylim(0, 5)
        ax.set_yticks(np.arange(1, 6))
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=11)
        plt.title(f'{title}\n(Lickert Scale of 1-5 with 1 being lowest ranking and 5 being highest ranking)', y=1.05, pad=20, fontsize=14)
    
    # Add legend with improved positioning and labels
    legend = plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    legend.set_title('Metrics' if not is_binary else 'Binary Metrics')
    
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
        
        # 4. Physical Components Analysis
        physical_cols = df.columns[[31, 33, 35, 37, 44]]
        create_radar_plot(df, physical_cols, 'Physical Components Analysis\nEvaluation of Importance of Each Physical Component', 'physical_radar.png', colors=physical_colors)
        print("Created physical components radar plot")
        
        # 5. Cyber Components Analysis
        cyber_cols = df.columns[46:52]
        create_radar_plot(df, cyber_cols, 'Cyber Components Analysis\nEvaluation of Importance of Each Cyber Component', 'cyber_radar.png', is_binary=True, colors=cyber_colors)
        print("Created cyber components radar plot")
        
        print("\nAnalysis complete. All files saved to output directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
