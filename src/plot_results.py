import matplotlib.pyplot as plt

from typing import List, Dict

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

    def plot_average_Lev(self, lev_dict: Dict) -> plt:
        x_vals = []
        y_vals = []

        for pages, (sum_lev, doc_count) in sorted(lev_dict.items()):
            x_vals.append(pages)
            avg_lev = sum_lev / doc_count
            y_vals.append(avg_lev)

        plt.figure(figsize=(16,9))
        plt.plot(x_vals, y_vals, marker="o", linestyle="--", color="blue")
        plt.title("Average Levenshtein vs number of pages")
        plt.xlabel("Number of pages in one document")
        plt.ylabel("Average Lev. distance")
        plt.grid(True)
        plt.savefig('avg_lev_per_pages.png')
        plt.close()