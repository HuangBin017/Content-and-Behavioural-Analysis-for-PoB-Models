import pandas as pd
import glob
import os

import csv

input_folder = "src/labeled_data/csv_files"
output_file = "src/labeled_data/combined.csv"

files = glob.glob(os.path.join(input_folder, "*.csv"))

with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.writer(outfile)

    # Write the correct header once
    writer.writerow(["author", "title", "label"])

    for file in files:
        print(f"Processing: {file}")

        with open(
            file,
            "r",
            encoding="utf-8",
            errors="replace",
            newline=""
        ) as infile:

            reader = csv.reader(infile)

            # Skip header
            next(reader, None)

            for row in reader:

                # Ignore completely empty rows
                if not row:
                    continue

                # Need at least author + title + label
                if len(row) < 3:
                    continue

                author = row[0]
                label = row[-1]

                # Everything between author and label is the title
                title = ", ".join(row[1:-1])

                writer.writerow([
                    author,
                    title,
                    label
                ])

print(f"Combined {len(files)} files into {output_file}")