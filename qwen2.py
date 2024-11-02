import torch
import json
import glob
import os
import time
import pandas as pd

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from data import Data
from json_handler import JsonHandler

class Qwen2(Data, JsonHandler):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.xlsx_path = "matching_dates_cleaned.xlsx"
        self.model_variant = "Qwen/Qwen2-VL-72B-Instruct"

        if self.device.type == "cuda":
            self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        elif self.device.type == "mps" or self.device.type == "cpu":
            self.cache_dir = "/Users/bk/Documents/Zajęcia (luty - czerwiec 2024)/Pracownia-problemowa/.cache"

        self.model = self.get_model()
        self.processor = self.get_processor()

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Qwen2"

    def get_model(self):
        if self.device.type == "cuda":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_variant,
                torch_dtype = torch.float16,
                attn_implementation = "flash_attention_2",
                device_map = "auto",
                cache_dir = self.cache_dir,
            )
            # do not use if device_map is set to "auto"
            # model.to(model.device)

            return model
        elif self.device.type == "mps" or self.device.type == "cpu":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_variant,
                torch_dtype = torch.bfloat16,
                device_map = "auto",
                cache_dir = self.cache_dir,
            )

            return model

    def get_processor(self, memory_save = True):
        if (self.device.type == "mps" or self.device.type == "cpu") and memory_save:
            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            processor = AutoProcessor.from_pretrained(
                self.model_variant,
                cache_dir = self.cache_dir, 
                min_pixels = min_pixels, 
                max_pixels = max_pixels,
            )

            return processor
        elif self.device.type == "cuda" and memory_save:
            min_pixels = 1280 * 28 * 28
            max_pixels = 1280 * 28 * 28
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
        pngs_paths: list[str] = self.get_pngs_path_from_folder()
        max_batch_threshold: int = 15
        matched_dates: pd.ExcelFile = pd.read_excel(self.xlsx_path)

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
                "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."
            })

            message = [
                {
                    "role": "user",
                    "content": current_message_content
                },
            ]

            dataset.append(message)

            # debug
            print(f"Dataset: {dataset}")
            print(type(dataset))

        return dataset

    def get_input_and_output(self, data: dict = None) -> list[str]:
        separate_outputs: list[str] = []

        for elem in data:
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

            generated_ids = self.model.generate(**inputs, max_new_tokens = 32768)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            separate_outputs.extend(output_text)

        # debug
        print(f"Separate outputs: {separate_outputs}")
        print(type(separate_outputs))

        return separate_outputs

    def save_json(self) -> None:
        generated_jsons = self.get_input_and_output(self.get_dataset())
        
        for i, output_text in enumerate(generated_jsons):
            try:
                self.json_dump(context = output_text, idx = i)
            except Exception as e:
                print(f"Error occurred in {self.save_json.__name__}, error: {e}")
