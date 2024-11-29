import torch
import json
import glob
import os
import time
import pandas as pd

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from data import Data
from json_handler import JsonHandler
from tqdm import tqdm
from utils import Utils
from qwen2half import Qwen2Half
class Qwen2(Data, Qwen2Half):
    def __init__(self, model = None) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xlsx_path = "matching_dates_cleaned.xlsx"
        self.model_variant = "Qwen/Qwen2-VL-72B-Instruct"

        if self.device.type == "cuda":
            self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        elif self.device.type == "mps" or self.device.type == "cpu":
            self.cache_dir = "/Users/bk/Documents/Zajęcia (luty - czerwiec 2024)/Pracownia-problemowa/.cache"

        self.model = model
        self.processor = self.get_processor()

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Qwen2"

    def get_custom_config(self) -> None:
        # custom quantization method
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            llm_int8_enable_fp32_cpu_offload = True
        )

        return quantization_config

    def get_model(self):
        # check if model already loaded version 2.0
        if hasattr(self, "model") and self.model is not None:
            print(f"Model {self.model_variant} already loaded, skipping...")
            return self.model
        
        if self.device.type == "cuda":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_variant,
                torch_dtype = torch.bfloat16,
                attn_implementation = "flash_attention_2",
                device_map = "auto",
                cache_dir = self.cache_dir,
                quantization_config = self.get_custom_config(), 
            )
            print(f"Loaded model {self.model_variant}")
            return model
        elif self.device.type == "mps" or self.device.type == "cpu":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_variant,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                cache_dir = self.cache_dir,
            )
            model.to(self.device)
            return model

    def get_processor(self, memory_save = True):
        if (self.device.type == "mps" or self.device.type == "cpu") and memory_save:
            min_pixels = 4 * 28 * 28
            max_pixels = 4 * 28 * 28
            processor = AutoProcessor.from_pretrained(
                self.model_variant,
                cache_dir = self.cache_dir, 
                min_pixels = min_pixels, 
                max_pixels = max_pixels,
            )

            return processor
        elif self.device.type == "cuda" and memory_save:
            min_pixels = 256 * 28 * 28
            max_pixels = 256 * 28 * 28
            processor = AutoProcessor.from_pretrained(
                self.model_variant,
                cache_dir = self.cache_dir, 
                min_pixels = min_pixels, 
                max_pixels = max_pixels,
            )

            return processor
        else:
            processor = AutoProcessor.from_pretrained(
                self.model_variant,
                cache_dir = self.cache_dir,
            )

            return processor

    def get_dataset(self) -> list[list[dict]]:
        dataset: list[list[dict]] = []
        max_batch_threshold: int = 5

        # convert to dataframe
        df: pd.ExcelFile = pd.read_excel(self.xlsx_path)
        dataframe = pd.DataFrame(df)

        # iterate over .xlsx for JSON and image folder paths
        for index, row in tqdm(dataframe.iterrows(), total = df.shape[0], desc = "Generowanie datasetu"):
            images_folder_path: str = row["Image folder path"]

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

                    dataset.append((message, subfolder_name))

                    # debug
                    # print(f"Dataset: {dataset}")
                    print(type(dataset))
                    print(len(dataset))
            else:
                continue

        return dataset

    def get_input_and_output(self, data: list[tuple[list[dict], str]]) -> list[tuple[str, str]]:
        separate_outputs: list[tuple[str, str]] = []

        for elem, subfolder_name in tqdm(data, desc = "Przetwarzanie danych"):
            text = self.processor.apply_chat_template(
                elem, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(elem)
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = self.model.generate(**inputs, max_new_tokens = 4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for output in output_text:
                separate_outputs.append((output, subfolder_name))

        # debug
        # print(f"Separate outputs: {separate_outputs}")
        print(type(separate_outputs))
        print(len(separate_outputs))

        return separate_outputs

    def save_json(self) -> None:
        Utils.delete_past_jsons()
        max_iter: int = 5
        generated_jsons = self.get_input_and_output(self.get_dataset())

        for i, (output_text, subfolder_name) in enumerate(tqdm(generated_jsons, desc = "Zapisywanie plików JSON")):
            for i in range(1, max_iter + 1):
                try:
                    self.json_dump(context = output_text, idx = i, subfolder = subfolder_name)
                    print(f"Json saved successfully!")
                    break
                except Exception as e:
                    print(f"Error occurred in {self.save_json.__name__}, error: {e}")

                    if i <= max_iter:
                        repaired_message: str = self.auto_repair_json(error_message = str(e), broken_json = output_text)

                        try:
                            self.json_dump(context = repaired_message, subfolder = subfolder_name)
                        except Exception as e:
                            print(f"Error occurred in {self.save_json.__name__}, error: {e}")
