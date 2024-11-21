from transformers import Trainer, TrainingArguments
from peft_lora import MyPeft

class MyTrainer:
    def __init__(self) -> None:
        self.training_params = None
        self.output_dir: str = "/net/scratch/hscra/plgrid/plgkruczek/.cache"
        self.log_dir: str = "/net/storage/pr3/plgrid/plgglemkin/isap/Magisterka-isap"
        self.trainer = None
        self.my_peft = MyPeft()

    def configure_training(self) -> None:
        self.training_params = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs = 5,
            logging_dir = self.log_dir,
        )

        return self.training_params

    def train(self) -> None:
        self.trainer = Trainer(
            model = self.my_peft.get_peft_model(),
            args = self.training_params,
        )

        return self.trainer