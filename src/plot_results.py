import matplotlib.pyplot as plt

from typing import List

class PlotResults:
    def __repr__(self) -> str:
        return "Class for plotting results"
    
    def plot_loss_function(self, train_loss: List[float], validation_loss: List[float]) -> plt:
        epochs = range(1, len(train_loss) + 1)

        plt.figure(figsize=(16, 9))
        plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
        plt.plot(epochs, validation_loss, 'r-o', label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('loss_plot.png')
        plt.close()
