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

    def json_load(self, path: str = None) -> str:
        colective_string: str = ""

        # for now only first 2 or 3 files, change to all files later
        for elem in glob.glob(f"{path}/*.json"):
            with open(elem, "r") as f:
                colective_string += f.read()

        return colective_string