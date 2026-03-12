import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_block_diagram(filename="block_diagram.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    
    # Define box styles
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e1f5fe', edgecolor='#0277bd', linewidth=2)
    process_props = dict(boxstyle='round,pad=0.5', facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=2)
    output_props = dict(boxstyle='round,pad=0.5', facecolor='#fce4ec', edgecolor='#c2185b', linewidth=2)
    
    # Coordinates
    x_center = 0.5
    y_input = 0.8
    y_process = 0.5
    y_output = 0.2
    
    # Input Layer
    ax.text(x_center, y_input, "Smartphone Sensors\n(GPS, Accelerometer, Gyro)", 
            ha='center', va='center', size=14, bbox=box_props)
            
    # Processing Layer
    ax.text(x_center, y_process, "Processing Unit\n(Feature Extraction + TFLite Model)", 
            ha='center', va='center', size=14, bbox=process_props)
            
    # Output Layer
    ax.text(0.3, y_output, "Haptic Alert\n(Vibration)", 
            ha='center', va='center', size=14, bbox=output_props)
    
    ax.text(0.7, y_output, "Visual Alert\n(UI Warning)", 
            ha='center', va='center', size=14, bbox=output_props)
            
    # Arrows
    # Input to Process
    ax.annotate("", xy=(x_center, y_process + 0.08), xytext=(x_center, y_input - 0.08),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
                
    # Process to Haptic
    ax.annotate("", xy=(0.3, y_output + 0.08), xytext=(x_center, y_process - 0.08),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
                
    # Process to Visual
    ax.annotate("", xy=(0.7, y_output + 0.08), xytext=(x_center, y_process - 0.08),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Block diagram saved to {filename}")

if __name__ == "__main__":
    create_block_diagram()
