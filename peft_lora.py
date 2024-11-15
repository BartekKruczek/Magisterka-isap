from peft import LoraConfig, TaskType, get_peft_model
from qwen2 import Qwen2

class MyPeft(Qwen2):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "Klasa do konfiguracji peft oraz wczytania modeli"
    
    def get_config(self) -> None:
        config = LoraConfig(
            task_type = TaskType.CAUSAL_LM,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            r = 8,
            lora_alpha = 32,
            lora_dropout = 0.1,
        )
        
        return config
    
    def get_peft_model(self) -> None:
        model = get_peft_model(self.get_model(), self.get_config())
        print(model)
        print(model.print_trainable_parameters())
        return model