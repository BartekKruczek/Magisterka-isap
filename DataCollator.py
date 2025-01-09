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
        zwracanych przez get_dataset().
        Zwracamy dict z polami np.:
        input_ids, attention_mask, pixel_values, labels, itp.
        """
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
            min_pixels = 512 * 28 * 28,
            max_pixels = 1024 * 28 * 28,
        )

        merged_texts = []
        image_inputs_list = []

        for (message, subfolder_name, json_ground_path) in examples:
            try:
                with open(json_ground_path, "r", encoding="utf-8") as f:
                    json_str = f.read()
            except:
                json_str = ""
                print(f"Error loading json file")
            
            # 2) Generujemy prompt (z tokenem [IMAGE] itd.) przez apply_chat_template
            prompt_str = processor.apply_chat_template(
                message,
                tokenize = False,
                add_generation_prompt = True
            )

            # 3) Sklejamy prompt + docelowe JSON w JEDNĄ sekwencję (teacher forcing)
            #    Tu można dodać dowolny separator, np. "\n", " Answer: ", itp.
            merged_text = prompt_str + "\n" + json_str
            merged_texts.append(merged_text)

            # 4) Wyciągamy obrazy
            image_inputs, _ = process_vision_info(message)
            image_inputs_list.append(image_inputs)

        # 5) Jednorazowo wywołujemy processor na CAŁYM batchu
        batch = processor(
            text=merged_texts,
            images=image_inputs_list,
            padding=True,
            return_tensors="pt",
        )

        # 6) Teacher forcing: labels = input_ids.clone()
        labels = batch["input_ids"].clone()

        # 7) Ustawiamy -100 na tokenach, których nie chcemy liczyć w cross-entropy:
        #    - [PAD]
        #    - tokeny obrazu (np. 151652, 151653, 151655)
        #    - ewentualnie prompt (jeśli chcesz go też pominąć)
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Jeśli w Qwen2-VL te ID odpowiadają tokenowi [IMAGE], maskujemy je też:
        potential_image_token_ids = [151652, 151653, 151655]
        for image_token_id in potential_image_token_ids:
            labels[labels == image_token_id] = -100

        # (opcjonalnie) maskowanie promptu - 
        #  jeśli prompt jest krótki i zawsze taki sam, możesz heurystycznie
        #  wykryć jego początkowe tokeny i zamienić na -100.

        batch["labels"] = labels
        return batch