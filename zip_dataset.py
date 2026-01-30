import shutil
import os

def zip_dataset():
    source_dir = "dataset_256"
    output_filename = "dataset_256"
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} does not exist. Did you run metadata preprocessing?")
        return

    print(f"Zipping '{source_dir}' to '{output_filename}.zip'...")
    shutil.make_archive(output_filename, 'zip', source_dir)
    print("Done! Upload 'dataset_256.zip' to your Google Drive.")

if __name__ == "__main__":
    zip_dataset()
