import os
import regex as re
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

from utils import Utils

class CustomDataSets:
    def __init__(self, model_id = "Qwen/Qwen2-VL-7B-Instruct"):
        self.model_id = model_id

    def process_years_to_excel(self, years_to_iterate, excel_filename="matched_dates_cleaned_version2.xlsx", debug: bool = True):
        base_json_path = "lemkin-json-from-html"
        json_results = []
        for year_val in years_to_iterate:
            json_folder_base = os.path.join(base_json_path, str(year_val))

            if debug:
                print(f"Przetwarzanie JSON w: {json_folder_base}", flush = True)

            for root, dirs, files in os.walk(json_folder_base):
                for file in files:
                    if file.endswith(".json"):
                        filename_no_ext = os.path.splitext(file)[0]
                        try:
                            year_str, doc_id_str = filename_no_ext.split("_", 1)
                            year = int(year_str)
                            doc_id = doc_id_str.strip()
                        except ValueError:
                            continue

                        match_json = re.search(r'(\d+)$', doc_id)
                        short_doc_id_json = str(int(match_json.group(1))) if match_json else doc_id

                        if len(short_doc_id_json) >= 4:
                            last4_doc_id_json = str(int(short_doc_id_json[-4:]))
                        else:
                            last4_doc_id_json = str(int(short_doc_id_json)) if short_doc_id_json.isdigit() else short_doc_id_json

                        full_path = os.path.join(root, file)
                        json_results.append({
                            "year": year,
                            "json_id": short_doc_id_json,
                            "last4_doc_id": last4_doc_id_json,
                            "json_path": full_path,
                        })
        
        if debug:
            print(f"Znaleziono {len(json_results)} plików JSON.", flush = True)

        base_pdf_path = "lemkin-pdf"
        png_results = []
        for year_val in years_to_iterate:
            pdf_folder_base = os.path.join(base_pdf_path, str(year_val))
            if not os.path.isdir(pdf_folder_base):

                if debug:
                    print(f"Folder nie istnieje: {pdf_folder_base}", flush = True)

                continue
            for root, dirs, files in os.walk(pdf_folder_base):
                for d in dirs:
                    if d.endswith("_png"):
                        folder_path = os.path.join(root, d)
                        
                        base_name = d[:-4] if d.endswith("_png") else d

                        match_png = re.search(r'(\d+)$', base_name)
                        short_doc_id_png = str(int(match_png.group(1))) if match_png else None
                        if short_doc_id_png and len(short_doc_id_png) >= 4:
                            last4_doc_id_png = str(int(short_doc_id_png[-4:]))
                        elif short_doc_id_png:
                            last4_doc_id_png = str(int(short_doc_id_png))
                        else:
                            last4_doc_id_png = None

                        png_results.append({
                            "year": year_val,
                            "png_id": short_doc_id_png,
                            "last4_doc_id": last4_doc_id_png,
                            "image_folder_path": folder_path,
                        })

        df_json = pd.DataFrame(json_results)
        df_png = pd.DataFrame(png_results)

        df_merged = pd.merge(df_json, df_png, on=["year", "last4_doc_id"], how="inner")

        df_result = df_merged[["year", "json_path", "image_folder_path"]]
        df_result = df_result.rename(columns={
            "json_path": "JSON file path",
            "image_folder_path": "Image folder path"
        })

        if debug:
            print("Przykładowe dopasowania:", flush = True)
            print(df_result.head(), flush = True)
            print(len(df_result), "dopasowań znaleziono.", flush = True)

        df_result.to_excel(excel_filename, index=False)

        if debug:
            print(f"Dane zapisane do {excel_filename}", flush = True)
        
        return df_result

    def read_excel(self) -> pd.DataFrame:
            return pd.read_excel("matched_dates_cleaned_version2.xlsx", engine = 'openpyxl')
        
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

        return sorted(pngs_paths, key=lambda path: self.sorting_by_page_number(path))

    def get_dataset(self, debug: bool = False, dataframe: pd.DataFrame = None) -> list[list[dict]]:
        dataset: list[list[dict]] = []
        max_batch_threshold: int = 10

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
                        "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Leave only generated structure, which will be dumped in the future"
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

        self.process_years_to_excel(years_to_iterate=[2014,2015,2016,2017,2018,2019,2020,2021])

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
            self.model_id,
            cache_dir="/net/scratch/hscra/plgrid/plgkruczek/.cache",
            min_pixels=128*28*28,
            max_pixels=256*28*28,
        )

        merged_texts = []
        image_inputs_list = []
        prompt_lengths = []

        for example in examples:
            message = example["messages"]
            json_ground_path = example["json_ground_path"]

            try:
                with open(json_ground_path, "r", encoding="utf-8") as f:
                    json_str = f.read()
            except Exception as e:
                json_str = ""
                print(f"Error loading json file: {json_ground_path}: {e}", flush=True)

            prompt_str = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )

            prompt_tokens = processor.tokenizer(prompt_str, return_tensors="pt")
            prompt_length = prompt_tokens.input_ids.size(1)
            prompt_lengths.append(prompt_length)

            merged_text = prompt_str + "\n" + json_str
            merged_texts.append(merged_text)

            image_inputs, _ = process_vision_info(message)
            image_inputs_list.append(image_inputs)

        for i, img in enumerate(image_inputs_list):
            if img is None:
                print(f"[DEBUG] image_inputs_list[{i}] is None!")
                
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

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        potential_image_token_ids = [151652, 151653, 151655]
        for image_token_id in potential_image_token_ids:
            labels[labels == image_token_id] = -100

        for i, prompt_len in enumerate(prompt_lengths):
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