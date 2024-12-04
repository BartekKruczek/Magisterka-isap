import torch
import pandas as pd
import os
import json

from json_handler import JsonHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

class Qwen2Half(JsonHandler):
    def __init__(self, model = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model_variant = "Qwen/Qwen2.5-7B-Instruct"
        self.xlsx_path = "matching_dates_cleaned.xlsx"

        if self.device.type == "cuda":
            self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        elif self.device.type == "mps" or self.device.type == "cpu":
            self.cache_dir = "/Users/bk/Documents/Zajęcia (luty - czerwiec 2024)/Pracownia-problemowa/.cache"

        self.model = model
        self.tokenizer = self.get_tokenizer()

    def __repr__(self) -> str:
        return "Klasa do obsługi modelu Qwen2.5"
    
    def get_custom_config(self) -> None:
        # custom quantization method
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            llm_int8_enable_fp32_cpu_offload = True
        )

        return quantization_config
    
    def get_model(self):
        if hasattr(self, "model") and self.model is not None:
            print(f"Model {self.model_variant} already loaded, skipping...")
            return self.model

        model = AutoModelForCausalLM.from_pretrained(
            self.model_variant,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
            cache_dir = self.cache_dir,
            attn_implementation = "flash_attention_2",
            quantization_config = self.get_custom_config(),
        )
        print(f"Loaded model {self.model_variant}")
        return model
    
    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_variant)

        return tokenizer
    
    def get_dataset(self, combined_string: str = None) -> list[dict]:
        df = pd.read_excel(self.xlsx_path)
        df_first_row = df.loc[0, "JSON file path"]

        prompt = f"Can you combine three separate json files into one? Here is example of json structure {df_first_row}. All files had been created from one law document. Text to combine: {combined_string}. Leave only generated structure."
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        return messages

    def combine_jsons_together(self, text_to_combine: str = None, json_ground_path: str = None) -> list[dict]:
        loaded_json: str = self.json_load(json_ground_path)

        prompt = f"Can you combine these json structures into one? Here is example of json structure {loaded_json}. All files had been created from one law document. Text to combine: {text_to_combine}. Leave only generated structure."
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    def get_response(self, messages: list[dict] = None) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens = 4096,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def get_response_training(self, messages: list[str] = None, debug: bool = True) -> str:
        if debug and not messages:
            raise ValueError("No messages provided for training response generation.")
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True
        )
        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens = 8192,
        )

        generated_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

        return response
    
    def save_combined_json(self) -> None:
        root_dir: str = "JSON_files"
        max_iterations: int = 3

        all_subdirs: list[str] = [os.path.join(root, dir) for root, dirs, _ in os.walk(root_dir) for dir in dirs]
        # debug
        print(all_subdirs)

        for dir_path in tqdm(all_subdirs, desc = "Przetwarzanie folderów JSON", unit = "folder"):
            # debug
            print(str(dir_path))
            number_json_files: int = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.json')]

            # if number of json files is more than 1 combine them
            if len(number_json_files) > 1:
                json_text = self.get_response(self.get_dataset(self.json_load(path = str(dir_path))))

                for i in range(1, max_iterations + 1):
                    try:
                        self.json_dump(json_text, idx = 999, subfolder = dir_path)
                        print(f"Combined json file saved successfully")
                        break
                    except Exception as e:
                        print(f"Error occurred in {self.save_combined_json.__name__}, error: {e}")

                        if i < max_iterations:
                            # if error occurred, we take response from model and try to save json again
                            repaired_attempt_message = self.auto_repair_json(error_message = str(e), broken_json = json_text)
                            json_text = self.get_response(repaired_attempt_message)

                            try:
                                self.json_dump(json_text, idx = 999, subfolder = dir_path)
                                print(f"Combined json file saved successfully")
                                break
                            except Exception as e:
                                print(f"Error occurred in {self.save_combined_json.__name__}, error: {e}")
            else:
                continue

    def make_json_from_generated_text(self, 
                                      generated_text: list[str] = None, 
                                      subfolder_name: str = None, 
                                      json_path: str = None,
                                      debug: bool = True) -> json:
        
        json_text_to_dump: str = self.get_response_training(self.get_dataset(
            text_to_combine = generated_text, 
            # json_ground_path = json_path,
            ))
        max_iterations: int = 3

        if debug:
            print(f"Type of json text to dump: {type(json_text_to_dump)} \n")
            print(f"Json text to dump: {json_text_to_dump}")

        for i in range(1, max_iterations + 1, 1):
            try:
                # json_text_to_dump = json.loads(json_text_to_dump)
                self.json_dump(json_text_to_dump, idx = 999, subfolder = subfolder_name)
                print(f"Combined json file saved successfully")
                return json_text_to_dump
            except Exception as e:
                print(f"Error occurred in {self.save_combined_json.__name__}, error: {e}")

                if i < max_iterations:
                    # if error occurred, we take response from model and try to save json again
                    repaired_attempt_message: str = self.auto_repair_json(error_message = str(e), broken_json = json_text_to_dump)
                    json_text: str = self.get_response(repaired_attempt_message)

                    try:
                        json_text = json.loads(json_text)
                        self.json_dump(json_text, idx = 999, subfolder = subfolder_name)
                        print(f"Combined json file saved successfully")
                        return json_text
                    except Exception as e:
                        print(f"Error occurred in {self.save_combined_json.__name__}, error: {e}")
