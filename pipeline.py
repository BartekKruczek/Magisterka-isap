from utils import Utils
from peft_lora import MyPeft
from trainer import MyTrainer
from metrics import CustomMetrics

class MyPipeLine:
    def __init__(self) -> None:
        self.utils = Utils
        self.my_peft = MyPeft()
        self.my_trainer = MyTrainer()
        self.my_metrics = CustomMetrics()

    def __repr__(self) -> str:
        return "Główna klasa do obsługi przepływu informacji z nacieskiem na trening modelu"
    
    def combine_all(self) -> None:
        pass