import csv
import glob
import os

input_folder = "src/labeled_data/csv_files"
output_folder = "src/labeled_data/fixed_csv"

os.makedirs(output_folder, exist_ok=True)

for input_file in glob.glob(os.path.join(input_folder, "*.csv")):
    output_file = os.path.join(
        output_folder,
        os.path.basename(input_file)
    )

    with open(input_file, "r", encoding="utf-8", errors="replace", newline="") as infile, \
         open(output_file, "w", encoding="utf-8", newline="") as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read header
        header = next(reader)

        # Keep only the first 3 columns in the output
        writer.writerow(header[:3])

        for row in reader:
            if len(row) < 3:
                continue

            first_column = row[0]
            last_column = row[-1]

            # Combine everything between first and last columns
            middle_columns = ", ".join(row[1:-1])

            writer.writerow([
                first_column,
                middle_columns,
                last_column
            ])

    print(f"Processed: {input_file}")
    
'''
• Deserving of reward: the content of the post is original or
highly informative, enriches the knowledge of those who
read it or otherwise contains very useful information for
the end reader. It undoubtedly deserves to be rewarded;
• Not deserving of reward: the content of the post contains
meaningless phrases or content that is totally irrelevant
to the end reader or is offensive and/or inappropriate,
or it is just spam, containing singular images and only a few words. Not deserving of a reward;

given these 2 criterias, label the dataset and output 1 csv file with 3 columns: author, title and label. you do not need to include the body and json metadata in your csv
'''