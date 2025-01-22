import pandas as pd
import os
import torch

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import Utils
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

class DataSets:
    def __init__(self, excel_file_path: str = None):
        self.excel_file_path = "matched_dates_cleaned_version2.xlsx"

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
        max_batch_threshold: int = 2

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

                    # dataset.append((message, subfolder_name, json_ground_path))
                    dataset.append({
                        "messages": message,
                        "subfolder_name": subfolder_name,
                        "json_ground_path": json_ground_path
                    })

                    if debug:
                        print(f"Dataset: {dataset}")
                        print(type(dataset))
                        print(len(dataset))
            else:
                continue

        return dataset
    
    def split_datasets(self):
        if not os.path.exists("test_csv/"):
            os.makedirs("test_csv/")

        df = self.read_excel()

        # split the dataset into training and validation
        train_df, tmp_df = train_test_split(df, test_size = 0.4, random_state = 42)

        # take 50% of the tmp_df for validation and test
        val_df, test_df = train_test_split(tmp_df, test_size = 0.5, random_state = 42)
        test_df.to_csv("test_csv/test_set.csv", index=False)

        # create dataset from the split
        train_dataset = self.get_dataset(dataframe = train_df)
        val_dataset = self.get_dataset(dataframe = val_df)
        test_df = self.get_dataset(dataframe = test_df)

        return train_dataset, val_dataset, test_df

    def collate_fn(self, examples, debug: bool = False):
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            cache_dir="/net/scratch/hscra/plgrid/plgkruczek/.cache",
            min_pixels=128*28*28,
            max_pixels=256*28*28,
        )
        # processor.to(torch.float16)

        merged_texts = []
        image_inputs_list = []
        prompt_lengths = []  # lista do przechowywania długości promptów

        for example in examples:
            message = example["messages"]
            json_ground_path = example["json_ground_path"]

            try:
                with open(json_ground_path, "r", encoding="utf-8") as f:
                    json_str = f.read()
            except Exception as e:
                json_str = ""
                print(f"Error loading json file: {json_ground_path}: {e}", flush=True)

            # Generujemy prompt
            prompt_str = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )

            # Obliczamy długość tokenów promptu
            prompt_tokens = processor.tokenizer(prompt_str, return_tensors="pt")
            prompt_length = prompt_tokens.input_ids.size(1)
            prompt_lengths.append(prompt_length)

            # Łączymy prompt z ground truth JSON-em
            merged_text = prompt_str + "\n" + json_str
            merged_texts.append(merged_text)

            # Wyciągamy obrazy
            image_inputs, _ = process_vision_info(message)
            image_inputs_list.append(image_inputs)

        # Tworzymy batch
        batch = processor(
            text=merged_texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )

        if debug:
            print(batch)
            print(type(batch))
            for key, value in batch.items():
                print(f"Key: {key}, Type: {type(value)}, Tensor dtype: {value.dtype}, Shape: {value.shape}")

        for key in batch:
            if key == 'pixel_values': 
                batch[key] = batch[key].to(torch.float16)

        # Tworzymy etykiety
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        potential_image_token_ids = [151652, 151653, 151655]
        for image_token_id in potential_image_token_ids:
            labels[labels == image_token_id] = -100

        # Maskujemy tokeny odpowiadające promptowi dla każdego przykładu
        for i, prompt_len in enumerate(prompt_lengths):
            # Upewniamy się, że prompt_len nie przekracza długości sekwencji
            seq_len = labels.size(1)
            if prompt_len < seq_len:
                labels[i, :prompt_len] = -100
            else:
                labels[i, :] = -100

        batch["labels"] = labels

        if debug:
            for key, value in batch.items():
                print(f"Key: {key}, Type: {type(value)}, Tensor dtype: {value.dtype}, Shape: {value.shape}")

        return batch