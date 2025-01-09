import pandas as pd
import os

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import Utils
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

class DataSets:
    def __init__(self, excel_file_path: str = None):
        self.excel_file_path = "matching_dates_cleaned.xlsx"

    def __repr__(self) -> str:
        return "DataSets class"
    
    def read_excel(self) -> pd.DataFrame:
        return pd.read_excel(self.excel_file_path, engine = 'openpyxl')
    
    def sorting_by_page_number(self, png_path: str = None) -> int:
        split1: str = png_path.split("/")[-1]
        split2 = split1.split(".")[0]
        split3 = split2.split("_")[-1]

        return int(split3)
    
    def get_pngs_path_from_folder(self, given_folder_path: str = None) -> list[str]:
        folder_path: str = str(given_folder_path)
        pngs_paths: list[str] = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".png"):
                    pngs_paths.append(os.path.join(root, file))

        return sorted(pngs_paths, key = self.sorting_by_page_number)
    
    def get_dataset(self, debug: bool = False, dataframe: pd.DataFrame = None) -> list[list[dict]]:
        dataset: list[list[dict]] = []
        max_batch_threshold: int = 5

        # convert to dataframe
        df = dataframe
        dataframe = pd.DataFrame(df)

        # iterate over .xlsx for JSON and image folder paths
        for _, row in tqdm(dataframe.iterrows(), total = df.shape[0], desc = "Generowanie datasetu"):
            images_folder_path: str = row["Image folder path"]
            json_ground_path: str = row["JSON file path"]

            # check if document is less than 10 pages, else skipp
            if Utils.check_length_of_simple_file(image_folder_path = images_folder_path):
                pngs_paths: list[str] = self.get_pngs_path_from_folder(given_folder_path = images_folder_path)
                # getting subfolder name
                subfolder_name: str = os.path.basename(images_folder_path)

                for i in range(0, len(pngs_paths), max_batch_threshold):
                    current_batch = pngs_paths[i:i+max_batch_threshold]
                    current_message_content: list[dict] = []

                    for current_image_path in current_batch:
                        root_image_path: str = os.path.relpath(current_image_path)
                        current_message_content.append({
                            "type": "image",
                            "image": root_image_path
                        })

                    current_message_content.append({
                        "type": "text",
                        "text": "Make a one, hierarchical .json from the image. Combine it with other messages."
                    })

                    message = [
                        {
                            "role": "user",
                            "content": current_message_content
                        },
                    ]

                    dataset.append((message, subfolder_name, json_ground_path))

                    if debug:
                        print(f"Dataset: {dataset}")
                        print(type(dataset))
                        print(len(dataset))
            else:
                continue

        return dataset
    
    def split_datasets(self):
        df = self.read_excel()

        # split the dataset into training and validation
        train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 42)

        # create dataset from the split
        train_dataset = self.get_dataset(dataframe = train_df)
        val_dataset = self.get_dataset(dataframe = val_df)

        return train_dataset, val_dataset

    def collate_fn(self, examples):
        """
        examples: lista krotek (message, subfolder_name, json_ground_path)
                zwracanych przez get_dataset()
        Zwracamy dict z polami np.:
        input_ids, attention_mask, pixel_values, labels, itp.
        """
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-72B-Instruct",
            cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
            min_pixels = 512 * 28 * 28,
            max_pixels = 1024 * 28 * 28,
        )

        messages = [ex[0] for ex in examples]
        json_paths = [ex[2] for ex in examples]

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs_list = []
        for msg in messages:
            image_inputs, _ = process_vision_info(msg)
            image_inputs_list.append(image_inputs)

        batch = processor(
            text=texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )

        labels_list = []
        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text_json = f.read()
                labels_list.append(text_json)
            except:
                labels_list.append("")
                print(f"Error while reading {path}")

        labels_enc = processor.tokenizer(
            labels_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        labels = labels_enc["input_ids"]

        labels[labels == processor.tokenizer.pad_token_id] = -100

        potential_image_token_ids = [151652, 151653, 151655]
        for image_token_id in potential_image_token_ids:
            labels[labels == image_token_id] = -100

        batch["labels"] = labels

        return batch