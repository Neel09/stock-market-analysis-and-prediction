import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.path import Path

def create_architecture_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Define box coordinates and sizes
    components = {
        'Data Sources': (0.1, 0.8, 0.2, 0.15),
        'Data Fetcher': (0.4, 0.8, 0.2, 0.15),
        'Data Processor': (0.7, 0.8, 0.2, 0.15),
        
        'Strategies': (0.1, 0.55, 0.2, 0.15),
        'Backtest Engine': (0.4, 0.55, 0.2, 0.15),
        'Performance Metrics': (0.7, 0.55, 0.2, 0.15),
        
        'Visualization': (0.4, 0.3, 0.2, 0.15),
        'Results Storage': (0.7, 0.3, 0.2, 0.15)
    }
    
    strategies = {
        'MA Crossover': (0.06, 0.45, 0.12, 0.06),
        'RSI': (0.22, 0.45, 0.12, 0.06),
        'MACD': (0.06, 0.37, 0.12, 0.06),
        'Bollinger Bands': (0.22, 0.37, 0.12, 0.06)
    }
    
    # Define colors
    colors = {
        'Data Sources': '#3498db',
        'Data Fetcher': '#3498db',
        'Data Processor': '#3498db',
        'Strategies': '#e74c3c',
        'Backtest Engine': '#2ecc71',
        'Performance Metrics': '#f39c12',
        'Visualization': '#9b59b6',
        'Results Storage': '#1abc9c'
    }
    
    strategy_color = '#e74c3c'
    arrow_color = '#34495e'
    
    # Draw boxes for main components
    for name, (x, y, width, height) in components.items():
        color = colors.get(name, '#cccccc')
        rect = mpatches.FancyBboxPatch((x, y), width, height,
                                       boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                                       ec=color, fc=color, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, name, ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
    
    # Draw strategy sub-boxes
    for name, (x, y, width, height) in strategies.items():
        rect = mpatches.FancyBboxPatch((x, y), width, height,
                                       boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                                       ec=strategy_color, fc=strategy_color, alpha=0.4)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, name, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    # Draw arrows
    arrows = [
        # Data flow
        ('Data Sources', 'Data Fetcher'),
        ('Data Fetcher', 'Data Processor'),
        ('Data Processor', 'Backtest Engine'),
        
        # Strategy to Backtest
        ('Strategies', 'Backtest Engine'),
        
        # Backtest to Performance Metrics
        ('Backtest Engine', 'Performance Metrics'),
        
        # Results flow
        ('Performance Metrics', 'Visualization'),
        ('Performance Metrics', 'Results Storage')
    ]
    
    for start, end in arrows:
        # Calculate start and end points
        start_box = components[start]
        end_box = components[end]
        
        # Determine direction and adjust start/end points
        if start_box[0] < end_box[0]:  # right arrow
            start_point = (start_box[0] + start_box[2], start_box[1] + start_box[3]/2)
            end_point = (end_box[0], end_box[1] + end_box[3]/2)
        elif start_box[0] > end_box[0]:  # left arrow
            start_point = (start_box[0], start_box[1] + start_box[3]/2)
            end_point = (end_box[0] + end_box[2], end_box[1] + end_box[3]/2)
        elif start_box[1] < end_box[1]:  # up arrow
            start_point = (start_box[0] + start_box[2]/2, start_box[1] + start_box[3])
            end_point = (end_box[0] + end_box[2]/2, end_box[1])
        else:  # down arrow
            start_point = (start_box[0] + start_box[2]/2, start_box[1])
            end_point = (end_box[0] + end_box[2]/2, end_box[1] + end_box[3])
        
        # Calculate control points for curved arrow
        if start == 'Strategies' and end == 'Backtest Engine':
            # Special case for strategies to backtest engine
            # Create a curved arrow using annotate instead of PathPatch
            ax.annotate('', 
                       xy=end_point, xycoords='data',
                       xytext=start_point, textcoords='data',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                      color=arrow_color, linewidth=2))
        else:
            # Regular arrow
            ax.annotate('', 
                       xy=end_point, xycoords='data',
                       xytext=start_point, textcoords='data',
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                                      color=arrow_color, linewidth=2))
    
    # Add ML and LSTM strategies with connection to Strategies
    ml_box = (0.06, 0.29, 0.12, 0.06)
    lstm_box = (0.22, 0.29, 0.12, 0.06)
    
    for name, coords in [('ML Strategy', ml_box), ('LSTM Strategy', lstm_box)]:
        x, y, width, height = coords
        rect = mpatches.FancyBboxPatch((x, y), width, height,
                                       boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                                       ec=strategy_color, fc=strategy_color, alpha=0.4)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, name, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')
    
    # Add UI component
    ui_box = (0.1, 0.05, 0.2, 0.15)
    ui_color = '#9b59b6'
    rect = mpatches.FancyBboxPatch(ui_box[:2], ui_box[2], ui_box[3],
                                   boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                                   ec=ui_color, fc=ui_color, alpha=0.6)
    ax.add_patch(rect)
    ax.text(ui_box[0] + ui_box[2]/2, ui_box[1] + ui_box[3]/2, 'Command Line UI', 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Add arrow from Visualization to UI
    viz_box = components['Visualization']
    viz_point = (viz_box[0] + viz_box[2]/2, viz_box[1])
    ui_point = (ui_box[0] + ui_box[2]/2, ui_box[1] + ui_box[3])
    
    ax.annotate('', 
               xy=ui_point, xycoords='data',
               xytext=viz_point, textcoords='data',
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', 
                              color=arrow_color, linewidth=2))
    
    # Set limits and hide axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add title
    ax.set_title('Nifty 50 Algorithmic Trading System Architecture', fontsize=16, pad=20)
    
    # Add legend for component types
    handles = []
    for component, color in [('Data Components', '#3498db'), 
                            ('Strategy Components', '#e74c3c'),
                            ('Backtest Engine', '#2ecc71'),
                            ('Analysis Components', '#f39c12'),
                            ('Visualization', '#9b59b6'),
                            ('Storage', '#1abc9c')]:
        patch = mpatches.Patch(color=color, alpha=0.6, label=component)
        handles.append(patch)
    
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=3, frameon=True, fancybox=True, shadow=True)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Architecture diagram saved as 'architecture_diagram.png'")
    plt.close()

if __name__ == "__main__":
    create_architecture_diagram() 