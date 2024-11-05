import multiprocessing as mp
import os
import re
from collections import defaultdict

# Function to count the total number of lines in a file
def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for _ in file)

# Function to find the best phi for a specific group (layer_idx, head_num)

# Constants for bounds (window)
lower_bound = 87.33654
upper_bound = 88.72283

# Function to find the best phi for a specific group (layer_idx, head_num)
def find_optimal_phi(qk_values):
    if not qk_values:
        return None, 0

    max_coverage = 0
    best_phi = None

    # Define potential phi range based on qk values and the given bounds
    min_phi = min(qk_values) - upper_bound
    max_phi = max(qk_values) + lower_bound
    #min_phi = min(qk_values) + lower_bound # found NA problem
    #max_phi = max(qk_values) - upper_bound # found NA problem
    #min_phi = min(qk_values) - upper_bound # TODO test
    #max_phi = max(qk_values) + lower_bound # TODO test

    # Start with the initial potential phi value
    phi = min_phi
    while phi <= max_phi:
        coverage = sum(1 for qk in qk_values if (phi - lower_bound) <= qk <= (phi + upper_bound))
        if coverage > max_coverage:
            max_coverage = coverage
            best_phi = phi
        
        phi += 0.1  # Increment phi by 0.1

    return best_phi, max_coverage, len(qk_values)
#    # Start at a middle position
#    max_phi = 0
#    max_coverage = 0
#    best_phi = None
#
#    # Check phi values in a reasonable range around 0
#    for phi in range(-100, 100):
#        coverage = sum(1 for qk in qk_values if (phi - 87.33654) <= qk <= (phi + 88.72283))
#        if coverage > max_coverage:
#            max_coverage = coverage
#            best_phi = phi
#
#    return best_phi, max_coverage, len(qk_values)

# Function to calculate coverage using average of min and max qk values
def calculate_average_coverage(qk_values):
    if not qk_values:
        return None, 0

    lower_bound = min(qk_values)
    upper_bound = max(qk_values)
    average_phi = (lower_bound + upper_bound) / 2

    # Calculate coverage for the average phi
    covered = sum(1 for qk in qk_values if (average_phi - 87.33654) <= qk <= (average_phi + 88.72283))

    return average_phi, covered

# Worker function to process each group of layer_idx and head_num
def process_group(args):
    (layer_idx, head_num), qk_values = args
    best_phi, best_coverage, total_qk = find_optimal_phi(qk_values)
    average_phi, average_coverage = calculate_average_coverage(qk_values)

    # Calculate the difference between best_phi and average_phi coverages
    coverage_diff = best_coverage - average_coverage

    return (layer_idx, head_num), {
        "best_phi": best_phi,
        "best_coverage": best_coverage,
        "total_qk": total_qk,
        "average_phi": average_phi,
        "average_coverage": average_coverage,
        "coverage_diff": coverage_diff  # Add the difference
    }

# Function to parse the file and group qk values by layer_idx and head_num
def parse_file(filename):
    layer_head_groups = defaultdict(list)
    total_lines = count_lines(filename)

    with open(filename, 'r') as file:
        processed_lines = 0
        for line in file:
            processed_lines += 1

            # Process the line
            match = re.search(r'<<<grid\[\d+, \d+, \d+\]block\[\d+, \d+, \d+\]>>> lane \d+ seq_len \d+ layer_idx (\d+) head_num (\d+) qk (-?\d+\.\d+)', line)
            if match:
                layer_idx = int(match.group(1))
                head_num = int(match.group(2))
                qk_value = float(match.group(3))
                layer_head_groups[(layer_idx, head_num)].append(qk_value)

            # Print progress for every 100,000 lines
            if processed_lines % 100000 == 0:
                percent_done = (processed_lines / total_lines) * 100
                print(f'\rParsing progress: {percent_done:.2f}% ({processed_lines}/{total_lines})', end='')

    print(f'\rDone parsing {filename} ({processed_lines}/{total_lines})')
    return layer_head_groups

# Main function to parse the file and find optimal phi for each group
def main(filename):
    # Initialize accumulators for overall best phi, average phi, and coverage diff
    total_best_phi = 0
    total_average_phi = 0
    total_coverage_diff = 0
    total_best_coverage = 0
    total_average_coverage = 0
    total_qk = 0

    # Parse the file
    layer_head_groups = parse_file(filename)

    # Use multiprocessing to find optimal phi for each group
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_group, layer_head_groups.items())

    # Collect results
    phi_results = {key: result for key, result in results}

    # Print the results in the desired format
    for layer_idx in range(32):
        output = []
        for head_num in range(32):
            key = (layer_idx, head_num)
            if key in phi_results:
                result = phi_results[key]

                # Accumulate the values for overall totals
                if result['best_phi'] is not None:
                    total_best_phi += result['best_phi']
                total_average_phi += result['average_phi']
                total_coverage_diff += result['coverage_diff']
                total_best_coverage += result['best_coverage']
                total_average_coverage += result['average_coverage']
                total_qk += result['total_qk']
                
                # Prepare individual output for each layer and head
                if result['best_phi'] is not None:
                    output.append(
                        f"Layer {layer_idx}, Head {head_num}: "
                        f"Best phi = {result['best_phi']:.6f}, Coverage = {result['best_coverage']}/{result['total_qk']}, "
                        f"Average phi = {result['average_phi']:.6f}, Average coverage = {result['average_coverage']}, "
                        f"Coverage difference = {result['coverage_diff']}"
                    )
                else:
                    output.append(
                        f"Layer {layer_idx}, Head {head_num}: "
                        f"Best phi = {result['best_phi']:.6f}, Coverage = {result['best_coverage']}/{result['total_qk']}, "
                        f"Average phi = {result['average_phi']:.6f}, Average coverage = {result['average_coverage']}, "
                        f"Coverage difference = {result['coverage_diff']}"
                    )

                #total_coverage_diff += result["coverage_diff"]
                #output.append(
                #    f"Layer {layer_idx}, Head {head_num}: "
                #    f"Best phi = {result['best_phi']:.6f}, Coverage = {result['best_coverage']}/{result['total_qk']}, "
                #    f"Average phi = {result['average_phi']:.6f}, Average coverage = {result['average_coverage']}, "
                #    f"Coverage difference = {result['coverage_diff']}"
                #)
                #output.append(f"Layer {layer_idx}, Head {head_num}: Best phi = {phi_results[key]['best_phi']}, Coverage = {phi_results[key]['coverage']}/{phi_results[key]['total_qk']}, Average phi = {phi_results[key]['average_phi']}, Average coverage = {phi_results[key]['average_coverage']}")
        print(" | ".join(output))

    # Calculate overall averages
    overall_best_phi = total_best_phi / (32 * 32)
    overall_average_phi = total_average_phi / (32 * 32)
    overall_coverage_diff = total_coverage_diff
    overall_best_coverage = total_best_coverage
    overall_average_coverage = total_average_coverage

    # Calculate percentages
    after_coverage_percent = (total_best_coverage / total_qk) * 100
    before_coverage_percent = (total_average_coverage / total_qk) * 100
    gain_coverage_percent = (total_coverage_diff / total_qk) * 100

    # Print the overall result for all layers and heads combined
    print(f"\nOverall across all layers and heads:")
    print(f"Best phi (average) = {overall_best_phi:.6f}, Total Best Coverage = {overall_best_coverage}/{total_qk}")
    print(f"Average phi (average) = {overall_average_phi:.6f}, Total Average Coverage = {overall_average_coverage}")
    print(f"Total Coverage Difference = {overall_coverage_diff}")

    # Print before, after and gain percentages
    print(f"\nAfter: {after_coverage_percent:.2f}% = {total_best_coverage}/{total_qk}")
    print(f"Before: {before_coverage_percent:.2f}% = {total_average_coverage}/{total_qk}")
    print(f"Gain: {gain_coverage_percent:.2f}% = {total_coverage_diff}/{total_qk}")

# Entry point of the program
if __name__ == "__main__":
#    time python jack_multiprocess_find_phi.py | tee phi_short_medium_summary_241004
#    filename = "qk_layer_head_medium_final.log" # phi_medium_summary_240927
#    filename = "qk_layer_head_short_final.log" # phi_short_summary_240927
    filename = "qk_layer_head_short_medium_final.log" # phi_short_medium_summary_240930 # phi_short_medium_summary_241004
    main(filename)

