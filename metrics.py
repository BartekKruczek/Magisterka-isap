from json_handler import JsonHandler

class CustomMetrics(JsonHandler):
    def __init__(self) -> None:
        super.__init__()

    def calculate_tree_edit_distance(self, json_generated_path: str = None, json_test_path: str = None) -> float:
        json_generated, json_test = self.json_load_TED(json_file1_path = json_generated_path, json_file2_path = json_test_path)

        # debug
        print(f'Json 1 as dictionary: {str(json_generated)} \n')
        print(f'Json 2 as dictionary: {str(json_test)} \n')