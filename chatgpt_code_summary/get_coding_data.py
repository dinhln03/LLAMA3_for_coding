from datasets import load_dataset
import os

from datasets import load_dataset
import os

def download_and_save_individual_contents(dataset_name, data_dir, split, field_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, data_dir=data_dir, split=split, ignore_verifications=True)

    # Specify the directory where you want to save the dataset
    dataset_path = os.path.join("data", dataset_name.replace("/", "_"))
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)


    # Path for the directory to save individual content files
    content_dir = os.path.join(dataset_path, field_name)
    if not os.path.exists(content_dir):
        os.makedirs(content_dir)

    # Iterate over the dataset and save each content in a separate file
    for index, item in enumerate(dataset):
        if field_name in item:
            content_file_path = os.path.join(content_dir, f"{field_name}_{index}.txt")
            with open(content_file_path, 'w', encoding='utf-8') as file:
                file.write(str(item[field_name]))
            print(f"Saved {field_name} of item {index} to {content_file_path}")

download_and_save_individual_contents("bigcode/the-stack-smol", data_dir="data/python", split="train", field_name="content")
