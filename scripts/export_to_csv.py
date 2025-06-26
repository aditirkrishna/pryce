# Convert outputs to CSV
import sys
import csv

if len(sys.argv) != 3:
    print("Usage: python export_to_csv.py input.txt output.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for line in infile:
        value = line.strip()
        if value:
            writer.writerow([value])
print(f"Exported {input_file} to {output_file}")
