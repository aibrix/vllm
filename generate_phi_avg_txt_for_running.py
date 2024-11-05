import re
# Input: human readable format
# Output: duca cpu code readable format

# In/Out files
#old
#input_file = "phi_short_240927___origin"
#output_file = "qk_max_hardcoded_values_short.txt"
#
#input_file = "phi_short_summary_240927"
#output_file = "qk_max_hardcoded_values_short.txt"
#
#input_file = "phi_medium_summary_240927"
#output_file = "qk_max_hardcoded_values_medium.txt"
# mix
#input_file = "phi_short_medium_summary_240930"
#output_file = "phi_hardcoded_values_short_medium_summary_240930.txt"
# mix2
input_file = "phi_short_medium_summary_241006"
best_phi_output_file = "best_phi_values_short_medium_summary_241006.txt"
avg_phi_output_file = "avg_phi_values_short_medium_summary_241006.txt"

def parse_and_write_to_files(input_file, best_phi_output_file, avg_phi_output_file):
    # Read input file
    with open(input_file, 'r') as file:
        input_data = file.read()

    # Match pattern for best_phi and avg_phi
    pattern = r"Layer \d+, Head \d+: Best phi = ([\d.-]+), Coverage = \d+/\d+, Average phi = ([\d.-]+), Average coverage = \d+"

    # Open two files to write best_phi and avg_phi separately
    with open(best_phi_output_file, 'w') as best_file, open(avg_phi_output_file, 'w') as avg_file:
        for match in re.finditer(pattern, input_data):
            best_phi = match.group(1)
            avg_phi = match.group(2)
            # Write best_phi to one file
            best_file.write(f"{best_phi}\n")
            # Write avg_phi to another file
            avg_file.write(f"{avg_phi}\n")

# Call the function
parse_and_write_to_files(input_file, best_phi_output_file, avg_phi_output_file)
print(f"Input: {input_file}")
print(f"Best phi output: {best_phi_output_file}")
print(f"Average phi output: {avg_phi_output_file}")
