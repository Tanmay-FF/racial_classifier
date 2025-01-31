import os

def save_file_paths_to_text(adir, output_file):
    """
    Recursively collects all file paths under a directory and saves them to a text file.

    Parameters:
    - adir: The directory to search.
    - output_file: The file where paths will be written.
    """
    # Open the output file in write mode
    count=0
    with open(output_file, 'w') as f:
        # Walk through the directory tree
        for root, _, files in os.walk(adir):
            for file in files:
                # Get the full file path
                file_path = os.path.join(root, file)
                if count<20257:
                    # Write the file path to the text file
                    f.write(file_path + " 0" + '\n')
                    count+=1

    print(f"File paths have been written to {output_file}")

# Directory to search and output file path
directory_to_search = r"T:\GAC\demographic_classifier\FaceArg-tightly-cropped\asian"# Replace with your directory path
output_text_file = r"T:\GAC\demographic_classifier\dataset-training\Fairface_UTK_downloaded_FaceArg\data\FaceArg_extra_asian.txt" # Replace with desired output file name

# Call the function
save_file_paths_to_text(directory_to_search, output_text_file)
