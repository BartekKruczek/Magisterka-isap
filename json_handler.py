import os
import json
import time
import glob
import pandas as pd

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
    
    def json_to_token_conversion(self, json_obj: json, keys_only: bool = True) -> list[str]:
        """
        Konwertuje obiekt JSON na listę tokenów. 
        - Jeśli keys_only = True, dodaje tylko nazwy kluczy i tokeny strukturalne. 
        - Jeśli keys_only = False, dodaje również wartości, jeśli nie są one zagnieżdżonymi strukturami.
        """

        created_tokens: list[str] = []

        def search_for_keys(my_obj):
            if isinstance(my_obj, dict):
                for key, value in my_obj.items():
                    created_tokens.append(f"<x_{key}>")
                    
                    if isinstance(value, (dict, list)):
                        search_for_keys(value)
                    else:
                        if not keys_only:
                            created_tokens.append(str(value))

                    created_tokens.append(f"</x_{key}>")

            elif isinstance(my_obj, list):
                created_tokens.append("<list>")
                for item in my_obj:
                    search_for_keys(item)
                created_tokens.append("</list>")
            else:
                if not keys_only:
                    created_tokens.append(str(my_obj))

        search_for_keys(json_obj)
        return created_tokens
    
    def get_special_tokens_json_ground_truth(self, debug: bool = False) -> list[str]:
        all_uniqe_tokens: set = set()

        try:
            df: pd.DataFrame = pd.read_excel("matching_dates_cleaned.xlsx", engine = "openpyxl")
            json_paths = df["JSON file path"]
        
            for elem in json_paths:
                if not os.path.exists(elem):
                    print(f"JSON file does not exist at path: {elem}")
                    continue

                if debug:
                    print(f"Processing {elem}")

                try:
                    with open(elem, "r", encoding = "utf-8") as f:
                        json_obj = json.load(f)
                
                    tokens = self.json_to_token_conversion(json_obj = json_obj, keys_only = True)
                
                    for t in tokens:
                        all_uniqe_tokens.add(t)
                except Exception as e:
                    print(f"Error processing JSON at {elem}: {e}")

        except Exception as e:
            print(f"Error occured {str(e)} in function {self.get_special_tokens_json_ground_truth.__name__}")
    
        return list(all_uniqe_tokens)
    
    def create_tree_from_tokens(self, tokens: list[str], start_index: int = 0, label: str = "root") -> None:
        """
        Here is function to create custom nodes from tokens structure.
        """
        my_node = CustomNodeCreator(label = label)
        i = start_index

        while i < len(tokens):
            token = tokens[i]

            # Sprawdź, czy to zamykający znacznik klucza lub listy
            if token.startswith("</x_") and token.endswith(">"):
                # Koniec aktualnego bloku obiektu
                return my_node, i + 1

            elif token == "</list>":
                # Koniec aktualnej listy
                return my_node, i + 1

            # Sprawdź, czy to otwierający znacznik klucza obiektu
            elif token.startswith("<x_") and token.endswith(">"):
                # Wyciągamy nazwę klucza
                key_name = token[3:-1]  # usuń '<x_' i '>'
                # Rekurencyjnie budujemy poddrzewo dla tego klucza
                child_node, new_index = self.create_tree_from_tokens(tokens, i + 1, label=key_name)
                my_node.add_children(child_node)
                i = new_index

            # Sprawdź, czy to początek listy
            elif token == "<list>":
                # Rekurencja dla listy
                child_node, new_index = self.create_tree_from_tokens(tokens, i + 1, label="list")
                my_node.add_children(child_node)
                i = new_index

            else:
                # Tu powinny trafiać wartości prymitywne (jeśli keys_only=False)
                # lub pomniejsze elementy nie będące strukturą (liście)
                value_node = CustomNodeCreator(label=token)
                my_node.add_children(value_node)
                i += 1

        return my_node, i