import os
import json
import time
import glob

from node_creator import CustomNodeCreator

class JsonHandler:
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return "Klasa do obsługi wszelakiej maści plików json"
    
    def json_dump(self, context: str = None, idx: int = None, subfolder: str = '', debug: bool = True) -> None:
        mkdir_root = "JSON_files"
        subfolder_path = os.path.join(mkdir_root, subfolder)
        
        if debug:
            print(str(subfolder_path))

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        my_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        cleaned_json_text = context.replace("```json\n", "").replace("```", "").strip()
        cleaned_json_text = " ".join(cleaned_json_text.split())
        cleaned_json_text = "".join(cleaned_json_text.splitlines())
        
        print(cleaned_json_text)

        try:
            json_obj = json.loads(cleaned_json_text)
            with open(f"{subfolder_path}/{my_time}_{idx}.json", "w", encoding='utf-8') as f:
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
    
    def auto_repair_json(self, error_message: str = None, broken_json: str = None) -> None:
        prompt: str = f"Json structure is not valid, because of {error_message}. Fix it {broken_json}. Polish language only."

        messages: list[dict] = [
            {"role": "user", "content": prompt}
        ]

        return messages
    
    def json_load_TED(self, json_file1_path: str = None, json_file2_path: str = None) -> str:
        try:
            with open(json_file1_path, 'r', encoding = 'utf-8') as file1:
                string1: str = file1.read()

            with open(json_file2_path, 'r', encoding = 'utf-8') as file2:
                string2: str = file2.read()

            # make dictionary from it
            json1: dict = json.loads(string1)
            json2: dict = json.loads(string2)

            return json1, json2
        except Exception as e:
            print(f'Error occured {e} in function {self.json_load_TED.__name__}')
            return None, None
        
    def create_tree_from_json_string(self, json: str = None, label: str = 'root') -> None:
        """
        Here is function to create custom nodes from json structure
        """
        my_node = CustomNodeCreator(label = label)

        # dictionary case
        if isinstance(json, dict):
            for key in json:
                child = self.create_tree_from_json_string(json[key], label = key)
                my_node.add_children(child)

        # list as key case
        elif isinstance(json, list):
            for idx, elem in enumerate(json):
                child = self.create_tree_from_json_string(elem, label = 'list_item')
                my_node.add_children(child)

        return my_node