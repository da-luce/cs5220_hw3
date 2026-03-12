import re
import matplotlib.pyplot as plt
import numpy as np

def parse_and_generate_plot(input_file, output_image):
    # Read the leaderboard content from the file
    try:
        with open(input_file, 'r') as f:
            data = f.read()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Helper function to extract simulation times from a specific block
    def extract_block_times(block_name):
        pattern = rf"--- {block_name} ---(.*?)--- END {block_name} ---"
        match = re.search(pattern, data, re.DOTALL)
        if match:
            # Find all floating point numbers following 'Simulation Time ='
            return [float(x) for x in re.findall(r"Simulation Time = ([\d.]+) seconds", match.group(1))]
        return []

    # Process counts are fixed at 64, 128, and 256 for this hardware configuration
    processes = np.array([64, 128, 256])
    
    # Extract times for 1M and 2M scales
    times_1m = extract_block_times("SCALE_1M")
    times_2m = extract_block_times("SCALE_2M")

    if len(times_1m) < 3 or len(times_2m) < 3:
        print("Error: Could not find enough timing data in the file. Check block names.")
        return

    # Calculate Ideal Scaling lines starting from the 64-process baseline
    ideal_1m = times_1m[0] * (processes[0] / processes)
    ideal_2m = times_2m[0] * (processes[0] / processes)

    # Create the Plot
    plt.figure(figsize=(10, 6))
    
    # Plot Actual Data (Solid lines with markers)
    plt.loglog(processes, times_1m, 'o-', label='1M Particles (Actual)', linewidth=2, markersize=8)
    plt.loglog(processes, times_2m, 's-', label='2M Particles (Actual)', linewidth=2, markersize=8)

    # Plot Ideal Scaling (Dashed/Dotted lines)
    plt.loglog(processes, ideal_1m, '--', color='gray', alpha=0.6, label='Ideal Scaling (1M)')
    plt.loglog(processes, ideal_2m, ':', color='gray', alpha=0.6, label='Ideal Scaling (2M)')

    # Labels and Styling
    plt.title('Strong Scaling Analysis: 1M vs 2M Particles', fontsize=14)
    plt.xlabel('Number of Processes (Log Scale)', fontsize=12)
    plt.ylabel('Simulation Time [seconds] (Log Scale)', fontsize=12)
    
    # Ensure specific process points are labeled on the x-axis
    plt.xticks(processes, processes)
    plt.minorticks_off()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(loc='best')
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_image)
    print(f"Success: Scaling plot saved as {output_image}")

# Execution
if __name__ == "__main__":
    parse_and_generate_plot('leaderboard-submission.out', 'strong_scaling_analysis.png')