import torch
import time

from utils import Utils
from data import Data
from peft_lora import MyPeft
from trainer import MyTrainer
from metrics import CustomMetrics
from qwen2 import Qwen2
from qwen2half import Qwen2Half
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForCausalLM

class MyPipeLine:
    def __init__(self) -> None:
        self.model_variant = "Qwen/Qwen2-VL-7B-Instruct"
        self.model_variant_half = "Qwen/Qwen2.5-7B-Instruct"
        self.cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        
        self.model_qwen2 = self.get_model_qwen2()
        self.model_qwen2half = self.get_model_qwen2half()

        self.my_utils = Utils
        self.my_data = Data()
        self.my_peft = MyPeft()
        self.my_trainer = MyTrainer()
        self.my_metrics = CustomMetrics()
        self.my_qwen2 = Qwen2(model = self.model_qwen2)
        self.my_qwen2half = Qwen2Half(model = self.model_qwen2half)
        self.model_to_train = self.my_peft.get_peft_model()

    def __repr__(self) -> str:
        return "Główna klasa do obsługi przepływu informacji z nacieskiem na trening modelu"
    
    def get_custom_config(self) -> None:
        # custom quantization method
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            llm_int8_enable_fp32_cpu_offload = False
        )
        return quantization_config
    
    def get_model_qwen2(self):
        # check if model already loaded version 2.0
        if hasattr(self, "model_qwen2") and self.model_qwen2 is not None:
            print(f"Model {self.model_variant} already loaded, skipping...")
            return self.model_qwen2
        
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
    
    def get_model_qwen2half(self):
        if hasattr(self, "model_qwen2half") and self.model_qwen2half is not None:
            print(f"Model {self.model_variant_half} already loaded, skipping...")
            return self.model_qwen2half

        model = AutoModelForCausalLM.from_pretrained(
            self.model_variant_half,
            torch_dtype = torch.bfloat16,
            device_map = "auto",
            cache_dir = self.cache_dir,
            attn_implementation = "flash_attention_2",
            quantization_config = self.get_custom_config(),
        )
        print(f"Loaded model {self.model_variant_half}")
        return model
    
    def train(self, model = None, optimizer = None, criterion = None, num_epochs: int = None, debug: bool = True) -> None:
        """
        Long story short we create jsons using Qwen2.0-VL, parse them using Qwen2.5-LM, calculate TED with base json from
        matching_dates_cleaned.xlsx and fine tune model
        """
        # init section
        self.my_utils.delete_past_jsons()
        iterations_to_save_json: int = 5
        dataset = self.my_qwen2.get_dataset()
        separate_outputs: list[tuple[str, str]] = []

        if debug:
            print(f"Len of dataset: {len(dataset)} \n")
            # print(f"Dataset: {dataset}")

        num_epochs: int = 1
        for epoch in range(1, 1 + num_epochs, 1):
            start_time: float = time.time()
            print(f"Epoch: {epoch}")
            self.model_to_train.train()
            total_loss: float = 0
            elem_counter: int = 0

            # iterate over dataset
            for elem, subfolder_name, json_ground_path in tqdm(dataset, desc = "Iter over dataset", leave = True, dynamic_ncols = True):
                elem_counter += 1
                print("\n")
                print("-----------------------------------", flush = True)
                print(f"Processing element number {elem_counter} of {len(dataset)} \n", flush = True)

                if debug:
                    print(f"Processing element: {elem} \n", flush = True)
                    print(f"Subfolder name: {subfolder_name} \n", flush = True)
                    print(f"Processing json ground path: {json_ground_path} \n", flush = True)

                # qwen2 section
                text = self.my_qwen2.processor.apply_chat_template(
                elem, 
                tokenize = False, 
                add_generation_prompt = True
                )
                image_inputs, video_inputs = process_vision_info(elem)
                inputs = self.my_qwen2.processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generated_ids = self.my_qwen2.model.generate(**inputs, max_new_tokens = 8192)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text: list[str] = self.my_qwen2.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                if debug:
                    # print(f"Generated output: {output_text}", flush = True)
                    print(f"Type od output: {type(output_text)}", flush = True)
                    print(f"Subfolder name: {subfolder_name}", flush = True)
                    print(f"Output text: {output_text}", flush = True)

                # qwen2half section
                dumped_text: str = self.my_qwen2half.make_json_from_generated_text(
                    generated_text = output_text, 
                    subfolder_name = subfolder_name,
                    json_path = json_ground_path,
                )

                if debug:
                    print(f"Dumped text: {dumped_text}")

                # metrics section
                calculated_TED: int = self.my_metrics.calculate_tree_edit_distance(
                    json_generated = dumped_text,
                    json_test_path = json_ground_path
                    )
                print(f"Calculated TED: {calculated_TED}", flush = True)

                for output in output_text:
                    separate_outputs.append((output, subfolder_name))

                if debug:
                    print(f"Type of separate outputs: {type(separate_outputs)}", flush = True)
                    print(f"Len of separate outputs: {len(separate_outputs)}", flush = True)

                end_time: float = time.time()
                elapsed_time: float = (end_time - start_time) / 60
                print(f"Time elapsed: {elapsed_time:.2} minutes", flush = True)