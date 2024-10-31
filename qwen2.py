import torch
import json
import glob
import os
import time

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

    def get_dataset(self) -> list[list[dict]]:
        dataset = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_0.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_1.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_2.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_3.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_4.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_5.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_6.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_7.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_8.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_9.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_10.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_11.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_12.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_13.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_14.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_15.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_16.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_17.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_18.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_19.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_20.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_21.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_22.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_23.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_24.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_25.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_26.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_27.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_28.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_29.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_30.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_31.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_32.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_33.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_34.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_35.png"},
                        {"type": "text", "text": "Make a one, hierarchical .json from image. Combine it with other messages. Polish language only."}
                    ],
                },
            ],
        ]
        return dataset

    def get_input(self) -> dict:
        processor = self.get_processor()
        dataset = self.get_dataset()
        text = processor.apply_chat_template(
            dataset, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(dataset)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    def get_outputs(self) -> list:
        processor = self.get_processor()
        model = self.get_model()
        inputs = self.get_input()

        generated_ids = model.generate(**inputs, max_new_tokens = 65536)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text
