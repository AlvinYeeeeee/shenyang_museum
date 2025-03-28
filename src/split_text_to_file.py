import os
import argparse
from create_explanations import read_file_line_by_line


def split_text_to_file(file_path: str, output_folder: str):
    """
    Split the text file into individual files and save them in the output folder.
    
    Args:
        file_path (str): Path to the text file to split
        output_folder (str): Path to the folder to save the split files
    """
    # Read the file line by line
    explainations = read_file_line_by_line(file_path)

    # Save each explanation to a separate file
    for explanation in explainations:
        # Create the output file path
        output_file_path = os.path.join(output_folder, f"{explanation.serial_number}.txt")

        # Write the explanation text to the output file
        with open(output_file_path, "w") as file:
            file.write(explanation.text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path",
                        type=str,
                        default="data/text/shenyang.txt")
    parser.add_argument("--output-dir",
                        type=str,
                        default="data/explanations")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    split_text_to_file(args.text_path, args.output_dir)


if __name__ == "__main__":
    main()
