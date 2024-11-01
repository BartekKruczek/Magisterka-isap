import os
import json
import time
import glob

class JsonHandler:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Klasa do obsługi wszelakiej maści plików json"
    
    def json_dump(self, context: str = None, idx: int = None) -> json:
        mkdir_root = "JSON_files"
        if not os.path.exists(mkdir_root):
            os.makedirs(mkdir_root)

        my_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        cleaned_json_text = context.replace("```json\n", "").replace("```", "").strip()
        cleaned_json_text = " ".join(cleaned_json_text.split())
        cleaned_json_text = "".join(cleaned_json_text.splitlines())
        
        print(cleaned_json_text)

        try:
            json_obj = json.loads(cleaned_json_text)
            with open(f"{mkdir_root}/{my_time}_{idx}.json", "w") as f:
                json.dump(json_obj, f, indent = 4, ensure_ascii = False)
        except Exception as e:
            print(f"Error occurred in {self.json_dump.__name__}, error: {e}")

    def auto_repair_json(self) -> None:
        # how many times should model try to repair json file on his own
        max_number_of_iterations: int = 10

        for i in range(max_number_of_iterations):
            pass

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