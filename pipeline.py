import torch
import time
import json
import numpy as np

from utils import Utils
from data import Data
from peft_lora import MyPeft
from trainer import MyTrainer
from metrics import CustomMetrics
from qwen2 import Qwen2
from qwen2half import Qwen2Half
from json_handler import JsonHandler
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM, AdamW
from torch.nn import CrossEntropyLoss


class MyPipeline:
    def __init__(self) -> None:
        self.model_variant = "Qwen/Qwen2-VL-72B-Instruct"
        self.model_variant_half = "Qwen/Qwen2.5-72B-Instruct"
        self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"

        self.model_qwen2 = self.load_model(self.model_variant, Qwen2VLForConditionalGeneration)
        self.model_qwen2half = self.load_model(self.model_variant_half, AutoModelForCausalLM)

        self.my_utils = Utils
        self.my_data = Data()
        self.my_json = JsonHandler()
        self.my_peft = MyPeft()
        self.my_trainer = MyTrainer()
        self.my_metrics = CustomMetrics()
        self.my_qwen2 = Qwen2(model=self.model_qwen2)
        self.my_qwen2half = Qwen2Half(model=self.model_qwen2half)
        self.model_to_train = self.my_peft.get_peft_model()

    def __repr__(self) -> str:
        return "Główna klasa do obsługi przepływu informacji z naciskiem na trening modelu"

    def get_custom_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )

    def load_model(self, model_name: str, model_class):
        if hasattr(self, "model_qwen2") and self.model_qwen2 is not None:
            print(f"Model {self.model_variant} already loaded, skipping...")
            return self.model_qwen2
        
        if hasattr(self, "model_qwen2half") and self.model_qwen2half is not None:
            print(f"Model {self.model_variant_half} already loaded, skipping...")
            return self.model_qwen2half

        try:
            print(f"Loading model {model_name}...")
            model = model_class.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                cache_dir=self.cache_dir,
                quantization_config=self.get_custom_config(),
            )
            print(f"Loaded model {model_name}")
            return model
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            raise

    def dataset_generator(self):
        """
        Generator that lazily yields dataset elements.
        """
        dataset = self.my_qwen2.get_dataset()
        for elem, subfolder_name, json_ground_path in dataset:
            yield elem, subfolder_name, json_ground_path

    def train(self, 
              model = None, 
              optimizer = None, 
              criterion = None, 
              num_epochs: int = 1, 
              do_generate_json_during_training: bool = False,
              do_dump_text: bool = False,
              debug: bool = True) -> None:
        
        self.my_utils.delete_past_jsons()
        separate_outputs = []
        
        model_train = self.model_to_train.train()
        optimizer = AdamW(params = model_train.parameters(), lr = 1e-5)
        criterion = CrossEntropyLoss

        # adding special tokens to tokenizer, only once before training
        all_special_tokens_from_ground_truth_dataset: list[str] = self.my_json.get_special_tokens_json_ground_truth()

        tokenizer = self.my_qwen2.processor.tokenizer
        tokenizer_2 = self.my_qwen2half.tokenizer

        num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": all_special_tokens_from_ground_truth_dataset})
        num_added_tokens = tokenizer_2.add_special_tokens({"additional_special_tokens": all_special_tokens_from_ground_truth_dataset})
        print(f"Added {num_added_tokens} special tokens.")

        self.model_qwen2.resize_token_embeddings(len(tokenizer))
        self.model_qwen2half.resize_token_embeddings(len(tokenizer_2), mean_resizing = False)

        for epoch in range(1, num_epochs + 1):
            print(f"Starting epoch {epoch}")
            start_time = time.time()
            elem_counter = 0

            for elem, subfolder_name, json_ground_path in tqdm(self.dataset_generator(), desc = "Processing dataset"):
                try:
                    elem_counter += 1
                    print("\n ----------------------", flush = True)
                    print(f"Processing element {elem_counter} \n", flush = True)

                    # Process inputs
                    text = self.my_qwen2.processor.apply_chat_template(
                        elem, 
                        tokenize = False, 
                        add_generation_prompt = True
                    )
                    image_inputs, video_inputs = process_vision_info(elem)
                    inputs = self.my_qwen2.processor(
                        text = text,
                        images = image_inputs,
                        videos = video_inputs,
                        padding = True,
                        return_tensors = "pt",
                    ).to("cuda")

                    if debug:
                        print(f"Inputs: {inputs}", flush = True)

                    if do_generate_json_during_training:
                        # Optimized token generation
                        generated_ids = self.my_qwen2.model.generate(
                            **inputs,
                            max_new_tokens = 2048,  # Limit the number of generated tokens
                            do_sample = False,  # Sampling for faster generation
                            temperature = 0.01,  # No randomness
                            use_cache = True  # Enable caching for faster computation
                        )
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = self.my_qwen2.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )

                        if debug:
                            print(f"Output text: {output_text}")

                        # Process Qwen2.5
                        if do_dump_text:
                            dumped_text = self.my_qwen2half.make_json_from_generated_text(
                                generated_text=output_text,
                                subfolder_name=subfolder_name,
                                json_path=json_ground_path,
                            )

                            # Calculate metrics
                            # calculated_TED = self.my_metrics.calculate_tree_edit_distance(
                            #     json_generated=dumped_text,
                            #     json_test_path=json_ground_path
                            # )
                            # print(f"Calculated TED: {calculated_TED}", flush=True)

                    # dumped text is combined json from one law document, here we convert it to tokens as well as ground truth json

                    # open ground truth json
                    if debug:
                        print(f"Processing {json_ground_path} file \n", flush = True)

                    try:
                        string1: str = ""
                        with open(json_ground_path, 'r', encoding = 'utf-8') as file1:
                            string1: str = file1.read()

                        json1: dict = json.loads(string1)

                        if debug:
                            print(f"Success in loading json object: {json1}", flush = True)
                    except Exception as e:
                        print(f"Error occured: {e}")

                    truth_tokens: list[str] = self.my_json.json_to_token_conversion(json_obj = json1, keys_only = False)
                    
                    if do_generate_json_during_training and do_dump_text:
                        # load dumped json
                        json_generated = json.load(dumped_text)
                        generated_tokens: list[str] = self.my_json.json_to_token_conversion(json_obj = json_generated)

                    if debug:
                        print(f"Json truth tokens: {truth_tokens}", flush = True)

                        if do_generate_json_during_training and do_dump_text:
                            print(f"Json generated tokens: {generated_tokens}", flush = True)

                    max_input_length = inputs.input_ids.shape[1]

                    labels = tokenizer(
                        truth_tokens,
                        is_split_into_words = True,
                        return_tensors = "pt",
                        max_length = max_input_length,
                        padding = "max_length",
                        truncation = True,
                    ).input_ids.to("cuda")

                    labels[labels == tokenizer.pad_token_id] = -100

                    if debug:
                        print(f"Shape inputs: {inputs.input_ids.shape}")
                        print(f"Shape labels: {labels.shape}")

                    outputs = model_train(**inputs)
                    logits = outputs.logits

                    loss = criterion((logits.view(-1, logits.size(-1)), labels.view(-1)))

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    print(f"Loss: {loss.item()}", flush = True)

                    # separate_outputs.append((output_text, subfolder_name))

                except Exception as e:
                    print(f"Error processing element {elem_counter}: {e}", flush=True)
                    continue

            elapsed_time = (time.time() - start_time) / 60
            print(f"Epoch {epoch} completed in {elapsed_time:.2f} minutes.")

        print("Training completed.")