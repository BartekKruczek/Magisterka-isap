import os
import json
import time

class JsonHandler:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Klasa do obsługi wszelakiej maści plików json"
    
    def json_dump(self, context: str = None) -> json:
        mkdir_root = "JSON_files"
        if not os.path.exists(mkdir_root):
            os.makedirs(mkdir_root)

        my_time = time.strftime("%Y-%m-%d_%H-%M-%S")

        try:
            with open(f"{mkdir_root}/{my_time}.json", "w") as f:
                json.dump(context, f, indent = 4, ensure_ascii = False)
        except Exception as e:
            print(f"Error occurred in {self.json_dump.__name__}, error: {e}")

    def auto_repair_json(self) -> None:
        # how many times should model try to repair json file on his own
        max_number_of_iterations: int = 10

        for i in range(max_number_of_iterations):
            pass