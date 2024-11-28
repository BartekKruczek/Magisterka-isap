import torch

from utils import Utils
from peft_lora import MyPeft
from trainer import MyTrainer
from metrics import CustomMetrics
from qwen2 import Qwen2

class MyPipeLine:
    def __init__(self) -> None:
        self.utils = Utils
        self.my_peft = MyPeft()
        self.my_trainer = MyTrainer()
        self.my_metrics = CustomMetrics()
        self.my_qwen2 = Qwen2()
        self.model_to_train = self.my_peft.get_peft_model()

    def __repr__(self) -> str:
        return "Główna klasa do obsługi przepływu informacji z nacieskiem na trening modelu"
    
    def train(self, model = None, optimizer = None, criterion = None, num_epochs: int = None, debug: bool = True) -> None:
        """
        Long story short we create jsons using Qwen2.0-VL, parse them using Qwen2.5-LM, calculate TED with base json from
        matching_dates_cleaned.xlsx and fine tune model
        """
        # init section
        self.utils.delete_past_jsons()
        iterations_to_save_json: int = 5
        dataset = self.my_qwen2.get_dataset()

        if debug:
            print(f"Len of dataset: {len(dataset)} \n")
            print(f"Dataset: {dataset}")

        num_epochs: int = 3
        for epoch in range(1, 1 + num_epochs, 1):
            print(f"Epoch: {epoch}")
            self.model_to_train.train()
            total_loss: float = 0
