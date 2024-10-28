import torch
import json
import glob
import os
import time

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from data import Data

class Qwen2(Data):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.xlsx_path = "matching_dates_cleaned.xlsx"
        self.model_variant = "Qwen/Qwen2-7B-Instruct"

        if self.device.type == "cuda":
            self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        elif self.device.type == "mps" or self.device.type == "cpu":
            self.cache_dir = "/Users/bk/Documents/Zajęcia (luty - czerwiec 2024)/Pracownia-problemowa/.cache"

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
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
            processor = AutoProcessor.from_pretrained(
                self.model_variant,
                cache_dir = self.cache_dir, 
                min_pixels = min_pixels, 
                max_pixels = max_pixels,
            )

            return processor
        elif self.device.type == "cuda" and memory_save:
            min_pixels = 256*28*28
            max_pixels = 1280*28*28
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

    def get_messages(self) -> list:
        df = self.get_xlsx_data(self.xlsx_path)

        # store images paths in a list
        images: list[str] = [
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_0.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_1.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_2.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_3.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_4.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_5.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_6.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_7.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_8.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_9.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_10.png",
        "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_11.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_12.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_13.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_14.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_15.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_16.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_17.png",
        # "lemkin-pdf/2014/WDU20140001826/T/D20141826TK_png/page_18.png",
        "lemkin-pdf/2014/WDU20140001589/O/D20141589_png/page_0.png",
        ]

        all_combined_image_contents = [{"type": "image", "image": image_path} for image_path in images]

        all_combined_other_contents = [
            {"type": "json", "json": df["JSON file path"].iloc[0]},
            {"type": "text", "text": "Can you make json from last image similar to what I gave you in json type and comparing structure to all another images? As output give me just json structure which can be dumped. Use polish language and letters as well."},
        ]

        messages = [
            {
            "role": "user",
            "content": all_combined_image_contents + all_combined_other_contents,
            }
        ]

        return messages

    def get_input(self) -> dict:
        processor = self.get_processor()
        messages = self.get_messages()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        return inputs

    def get_outputs(self) -> list:
        processor = self.get_processor()
        model = self.get_model()
        input = self.get_input()

        generated_ids = model.generate(**input, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text

    def create_json(self) -> json:
        mkdir_root = "JSON_files"
        if not os.path.exists(mkdir_root):
            os.makedirs(mkdir_root)

        my_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        for elem in self.get_outputs():
            cleaned_text = elem.replace("```json\n", "").replace("```", "").strip()

            # usuwanie nadmiarowych spacji i znaków nowej linii
            cleaned_text = " ".join(cleaned_text.split())
            cleaned_text = "".join(cleaned_text.splitlines())
            
            print(cleaned_text)
            
            try:
                json_obj = json.loads(cleaned_text)
                with open(f"./JSON_files/{my_time}.json", "w", encoding = "utf-8") as f:
                    json.dump(json_obj, f, indent = 4, ensure_ascii = False)
                print("JSON file saved successfully!")
            except json.JSONDecodeError as e:
                print("JSON decoding error:", e)
                self.create_txt(text = cleaned_text, error = str(e))
            except Exception as e:
                print("Error saving JSON file:", e)
                self.create_txt(text = cleaned_text, error = str(e))

        # self.clear_cache_memory()

    def auto_repair_json_QWEN(self) -> str:
        # get the newest .txt file from To_repair/txt_files directory
        txt_files = glob.glob("To_repair/txt_files/*.txt")
        latest_txt_file = max(txt_files, key = os.path.getctime)
        print("Latest txt file:", latest_txt_file)

        # load file: text message separated from error message using <SEP> separator
        with open(latest_txt_file, "r", encoding = "utf-8") as f:
            content = f.read()
            text, error_message = content.split("<SEP>")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "The following JSON is invalid and cannot be parsed."
                            f"The error message is: {error_message}. "
                            "Please correct the JSON so that it is valid and can be parsed. Leave the language as Polish."
                            "The invalid JSON is:\n```json\n"
                            f"{text}\n```"
                        ),
                    },
                ],
            }
        ]
        
        # sekcja odpowiedzialna za przetworzenie danych wejściowych
        processor = self.get_processor()
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # sekcja odpowiedzialna za generowanie poprawionego JSON-a
        model = self.get_model()
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        corrected_json_text = output_text[0]

        # oczyszczenie JSON-a z nadmiarowych spacji i znaków nowej linii
        corrected_json_text = corrected_json_text.replace("```json\n", "").replace("```", "").strip()
        corrected_json_text = " ".join(corrected_json_text.split())
        corrected_json_text = "".join(corrected_json_text.splitlines())
        
        # generowanie poprawionego JSON-a
        try:
            json_obj = json.loads(corrected_json_text)
            with open("output.json", "w", encoding = "utf-8") as f:
                json.dump(json_obj, f, indent = 4, ensure_ascii = False)
            print("JSON repaired file saved successfully!")
            print(f"Json: {corrected_json_text}")
        except json.JSONDecodeError as e:
            print("JSON repaired decoding error:", e)
            print(f"Json: {corrected_json_text}")
        except Exception as e:
            print("Error repair saving JSON file:", e)
            print(f"Json: {corrected_json_text}")

        self.clear_cache_memory()
