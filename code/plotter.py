import matplotlib.pyplot as plt
import os

output_dir = "plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save the figures in output dir
def plot_loss(history, model_path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and test loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_path}_loss_curve.png"), dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

def plot_acc(history, model_path):
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and test accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_path}_accuracy_curve.png"), dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()