import matplotlib.pyplot as plt
import torch

def show_feature_maps(model, loader, class_names, device):
    model.eval()
    images, labels = next(iter(loader))
    image = images[0:1].to(device)
    label = labels[0].item()

    with torch.no_grad():
        _, activations = model(image, return_activations=True)

    plt.figure(figsize=(3, 3))
    plt.imshow(image[0, 0].cpu().numpy(), cmap="gray")
    plt.title(f"Input image\n{class_names[label]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

    # to modify
    layers_to_show = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "conv3", "relu3", "pool3"]

    for layer_name in layers_to_show:
        feat = activations[layer_name][0].detach().cpu()
        
        # only the first 8 representations because of difficulty to plot
        n_channels = min(8, feat.shape[0])

        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        axes = axes.flatten()

        for i in range(8):
            axes[i].axis("off")

        for i in range(n_channels):
            axes[i].imshow(feat[i].numpy(), cmap="gray")
            axes[i].set_title(f"ch {i}")
            axes[i].axis("off")

        fig.suptitle(f"Feature maps at layer: {layer_name}", y=0.98)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        
def show_representation_summaries(model, loader, device):
    model.eval()
    images, labels = next(iter(loader))
    image = images[0:1].to(device)

    with torch.no_grad():
        _, activations = model(image, return_activations=True)

    summary_vectors = {}

    for name, tensor in activations.items():
        t = tensor[0].detach().cpu()

        if t.ndim == 3:
            summary_vectors[name] = t.mean(dim=(1, 2))
        elif t.ndim == 1:
            summary_vectors[name] = t

    for name, vec in summary_vectors.items():
        vec = vec.numpy()
        shown = vec[:40]

        plt.figure(figsize=(8, 3.5))
        plt.bar(range(len(shown)), shown)
        plt.xlabel("Representation dimension")
        plt.ylabel("Activation value")
        plt.title(f"Representation summary at layer: {name}")
        plt.tight_layout()
        plt.show()
        plt.close()
