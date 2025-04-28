import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_gantt_chart(results):
    """
    Create a Gantt chart showing the schedule of activities for each project
    """
    num_projects = results['raw_data']['NUM_project']
    num_activities = results['raw_data']['NUM_act']
    
    fig, axes = plt.subplots(num_projects, 1, figsize=(15, num_projects*5), sharex=True)
    
    if num_projects == 1:
        axes = [axes]
    
    # Custom colormap
    colors = ["#2980b9", "#3498db", "#1abc9c", "#16a085", "#27ae60", "#2ecc71", "#f1c40f", 
              "#f39c12", "#e67e22", "#d35400", "#e74c3c", "#c0392b", "#9b59b6", "#8e44ad"]
    
    for j in range(num_projects):
        ax = axes[j]
        
        # Sort activities by start time
        activities = [(i, results['start_times'][i][j], results['finish_times'][i][j]) 
                      for i in range(num_activities)]
        activities.sort(key=lambda x: x[1])  # Sort by start time
        
        # Plot each activity
        for idx, (i, start, finish) in enumerate(activities):
            color_idx = i % len(colors)
            ax.barh(f"Activity {i+1}", finish - start, left=start, height=0.5, 
                    color=colors[color_idx], alpha=0.8, edgecolor='black')
            
            # Add text labels
            ax.text(start + (finish - start) / 2, f"Activity {i+1}", f"{int(start)}-{int(finish)}", 
                   ha='center', va='center', color='black', fontweight='bold')
        
        # Add target line
        target = results['raw_data']['Target'][j]
        ax.axvline(x=target, color='red', linestyle='--', linewidth=2, 
                  label=f'Target ({target})')
        
        # Add completion time line
        completion = results['completion_times'][j]
        ax.axvline(x=completion, color='green', linestyle='-', linewidth=2, 
                  label=f'Completion ({int(completion)})')
        
        # Add labels and title
        ax.set_ylabel('Activities')
        ax.set_title(f'Project {j+1} Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    
    axes[-1].set_xlabel('Time (days)')
    plt.tight_layout()
    plt.savefig('gantt_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gantt chart saved as 'gantt_chart.png'")

def plot_resource_usage(results):
    """
    Create a visualization of the resource usage over time for each project
    """
    num_projects = results['raw_data']['NUM_project']
    num_raw_mat = results['raw_data']['NUM_raw_mat']
    num_sup = results['raw_data']['NUM_sup']
    
    # Process material allocations
    material_usage = np.zeros((num_projects, num_raw_mat))
    
    for o in range(len(results['material_allocations'])):
        for k in range(num_raw_mat):
            for s in range(num_sup):
                for j in range(num_projects):
                    material_usage[j, k] += results['material_allocations'][o][k][s][j]
    
    # Create dataframe for easier plotting
    df = pd.DataFrame(material_usage)
    df.columns = [f'Material {k+1}' for k in range(num_raw_mat)]
    df.index = [f'Project {j+1}' for j in range(num_projects)]
    
    # Plot
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(df, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=.5, 
                    cbar_kws={'label': 'Quantity'})
    
    plt.title('Raw Material Usage by Project', fontsize=16, fontweight='bold')
    plt.ylabel('Project')
    plt.xlabel('Material Type')
    plt.tight_layout()
    plt.savefig('resource_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Resource usage chart saved as 'resource_usage.png'")

def plot_project_delays(results):
    """
    Create a visualization comparing target dates vs. actual completion dates for projects
    """
    projects = [f'Project {j+1}' for j in range(results['raw_data']['NUM_project'])]
    finish_dates = [results['completion_times'][j] for j in range(results['raw_data']['NUM_project'])]
    target_dates = [results['raw_data']['Target'][j] for j in range(results['raw_data']['NUM_project'])]
    delays = [max(0, finish_dates[j] - target_dates[j]) for j in range(results['raw_data']['NUM_project'])]
    penalties = [delays[j] * results['raw_data']['Penalty'][j] for j in range(results['raw_data']['NUM_project'])]
    
    # Reduced figure size
    fig, ax2 = plt.subplots(figsize=(10, 6))
    
    # Create a twin axis for penalty values
    ax2_twin = ax2.twinx()
    
    # Create the grouped bar chart
    x = np.arange(len(projects))
    width = 0.35
    
    # Plot bars
    ax2.bar(x - width/2, target_dates, width, label='Target Date', color='#3498db', alpha=0.7)
    ax2.bar(x + width/2, finish_dates, width, label='Actual Completion', color='#2ecc71', alpha=0.7)
    # Plot delays and penalties as markers
    ax2.scatter(x, delays, marker='o', color='#d35400', s=100, label='Delay (days)', zorder=5)
    ax2_twin.scatter(x + width, penalties, marker='s', color='#8e44ad', s=100, label='Penalty (NIS)', zorder=5)
    
    # Add labels and annotations
    for i, delay in enumerate(delays):
        ax2.text(i, delay + 0.5, f'{int(delay)}', ha='center', va='bottom', color='#d35400')
    
    # Set a reasonable offset for the penalty label to avoid extreme values
    for i, penalty in enumerate(penalties):
        offset = min(500, penalty * 0.05)  # Reduced offset to 5% of penalty value, still capped at 500
        ax2_twin.text(i + width, penalty + offset, f'{int(penalty)}', ha='center', va='bottom', color='#8e44ad')
    
    ax2.set_ylabel('Delay (days)')
    ax2_twin.set_ylabel('Penalty (NIS)')
    ax2.set_title('Project Delays and Penalties', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(projects)
    ax2.grid(True, alpha=0.3)
    
    # Add both legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Skip tight_layout which can cause issues with large plots
    # plt.tight_layout()
    
    # Further reduced DPI to avoid size issues
    plt.savefig('project_delays.png', dpi=80, bbox_inches='tight')
    plt.close()
    print("Project delays chart saved as 'project_delays.png'")

def plot_supplier_allocation(results):
    """
    Create a visualization showing the allocation of raw materials from suppliers to projects
    """
    num_projects = results['raw_data']['NUM_project']
    num_raw_mat = results['raw_data']['NUM_raw_mat']
    num_sup = results['raw_data']['NUM_sup']
    
    # We'll create one plot per material type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Custom colors for suppliers
    supplier_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for k in range(num_raw_mat):
        ax = axes[k]
        
        # Calculate total allocation per supplier for this material
        sup_alloc = np.zeros((num_sup, num_projects))
        
        for o in range(len(results['material_allocations'])):
            for s in range(num_sup):
                for j in range(num_projects):
                    sup_alloc[s, j] += results['material_allocations'][o][k][s][j]
        
        # Create stacked bar chart
        bottoms = np.zeros(num_projects)
        for s in range(num_sup):
            ax.bar(range(num_projects), sup_alloc[s], bottom=bottoms, 
                   label=f'Supplier {s+1}', color=supplier_colors[s], alpha=0.7)
            
            # Add text labels for non-zero allocations
            for j in range(num_projects):
                if sup_alloc[s, j] > 0:
                    ax.text(j, bottoms[j] + sup_alloc[s, j]/2, f'{int(sup_alloc[s, j])}', 
                           ha='center', va='center', color='black', fontweight='bold')
            
            bottoms += sup_alloc[s]
        
        # Add project demand line
        for j in range(num_projects):
            ax.axhline(y=results['raw_data']['Quantity'][k][j], color='red', linestyle='--', 
                      alpha=0.7, xmin=j/num_projects, xmax=(j+1)/num_projects)
            ax.text(j, results['raw_data']['Quantity'][k][j] + 20, 
                   f'Demand: {results["raw_data"]["Quantity"][k][j]}', 
                   ha='center', va='bottom', color='red')
        
        ax.set_title(f'Material {k+1} Allocation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Quantity')
        ax.set_xticks(range(num_projects))
        ax.set_xticklabels([f'Project {j+1}' for j in range(num_projects)])
        ax.grid(True, alpha=0.3)
        
        # Only add legend to the first subplot
        if k == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('supplier_allocation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Supplier allocation chart saved as 'supplier_allocation.png'")

def plot_critical_path(results):
    """
    Visualize the critical path for each project
    """
    num_projects = results['raw_data']['NUM_project']
    num_activities = results['raw_data']['NUM_act']
    
    # We'll create one visualization per project
    for j in range(num_projects):
        # Identify critical activities (those with no slack)
        critical_activities = []
        
        # Extract start and finish times
        start_times = [results['start_times'][i][j] for i in range(num_activities)]
        finish_times = [results['finish_times'][i][j] for i in range(num_activities)]
        
        # For simplicity, we'll consider the path of activities that leads to the latest finish time
        # A more accurate approach would require analyzing the precedence relationships more carefully
        activities_by_finish = [(i, finish_times[i]) for i in range(num_activities)]
        activities_by_finish.sort(key=lambda x: x[1], reverse=True)
        
        latest_finish = activities_by_finish[0][1]
        for i, finish in activities_by_finish:
            if finish == latest_finish:
                critical_activities.append(i)
                latest_finish = start_times[i]
                
                # Find predecessor with matching finish time
                for i_pred in range(num_activities):
                    if finish_times[i_pred] == start_times[i]:
                        critical_activities.append(i_pred)
                        latest_finish = start_times[i_pred]
        
        # Remove duplicates and sort by start time
        critical_activities = list(set(critical_activities))
        critical_activities.sort(key=lambda i: start_times[i])
        
        # Create visualization
        plt.figure(figsize=(14, 6))
        
        # Plot all activities
        for i in range(num_activities):
            plt.barh(f'Activity {i+1}', finish_times[i] - start_times[i], left=start_times[i], 
                    color='#bdc3c7', height=0.5, alpha=0.4)
        
        # Highlight critical activities
        for i in critical_activities:
            plt.barh(f'Activity {i+1}', finish_times[i] - start_times[i], left=start_times[i], 
                    color='#e74c3c', height=0.5, alpha=0.8, edgecolor='black')
            
            plt.text(start_times[i] + (finish_times[i] - start_times[i])/2, f'Activity {i+1}', 
                   f'{int(start_times[i])}-{int(finish_times[i])}', 
                   ha='center', va='center', color='black', fontweight='bold')
        
        plt.title(f'Critical Path for Project {j+1}', fontsize=14, fontweight='bold')
        plt.xlabel('Time (days)')
        plt.ylabel('Activities')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.barh(0, 0, color='#bdc3c7', alpha=0.4, label='Regular Activity')
        plt.barh(0, 0, color='#e74c3c', alpha=0.8, label='Critical Activity')
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'critical_path_project_{j+1}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Critical path for Project {j+1} saved as 'critical_path_project_{j+1}.png'")
