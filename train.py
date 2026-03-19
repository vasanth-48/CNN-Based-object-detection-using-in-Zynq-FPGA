import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.resnet import ResNet, BasicBlock
from PIL import Image
import os

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ---------- ResNet-14 Definition ----------


def resnet14(num_classes=10):
    """A smaller ResNet: layers=[1,1,1,1] gives ~14 layers total."""
    return ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=num_classes)
# -----------------------------------------


def load_trained_model():
    """Load the trained ResNet-14 model for CIFAR-10."""
    model = resnet14(num_classes=10)
    model.load_state_dict(torch.load(
        "resnet14_cifar10.pth", map_location="cpu"))
    model.eval()
    return model


def predict_single_image(model, image_path):
    """Predict CIFAR-10 class for a single image file."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(output, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return {
        "predicted_class": CIFAR10_CLASSES[pred_idx],
        "class_index": pred_idx,
        "confidence": confidence,
        "raw_output": output[0].tolist(),
        "all_probabilities": probs[0].tolist()
    }


def test_on_cifar_samples():
    """Run quick predictions on the first 10 CIFAR-10 test images."""
    print("Testing on CIFAR-10 test samples…")
    model = load_trained_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=False, transform=transform)

    correct = 0
    total = 10
    print("\nPredictions on first 10 test images:")
    print("-" * 70)
    for i in range(total):
        image, true_label = test_dataset[i]
        image_batch = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image_batch)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(output, dim=1).item()
            conf = probs[0][pred_idx].item()

        pred_class = CIFAR10_CLASSES[pred_idx]
        true_class = CIFAR10_CLASSES[true_label]
        is_correct = pred_idx == true_label
        if is_correct:
            correct += 1
        mark = "✓" if is_correct else "✗"
        print(f"Image {i+1:2d}: {mark} Predicted: {pred_class:10s} "
              f"(True: {true_class:10s}) Confidence: {conf:.2f}")

        if i == 0:
            print("  Top 3 predictions for first image:")
            top3 = torch.topk(probs[0], 3).indices
            for j, idx in enumerate(top3):
                print(
                    f"    {j+1}. {CIFAR10_CLASSES[idx]:10s}: {probs[0][idx]:.3f}")

    acc = correct / total * 100
    print(f"\nAccuracy on {total} samples: {acc:.1f}%")


def analyze_model_output():
    """Print raw logits and probabilities for one CIFAR-10 test sample."""
    print("\nModel Output Analysis:")
    print("=" * 50)
    model = load_trained_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root="./data", train=False,
                                    download=False, transform=transform)
    image, true_label = test_dataset[0]
    image_batch = image.unsqueeze(0)

    with torch.no_grad():
        raw_output = model(image_batch)
        probs = torch.softmax(raw_output, dim=1)

    print(f"True class: {CIFAR10_CLASSES[true_label]}")
    print("\nRaw model output (logits):")
    for i, logit in enumerate(raw_output[0]):
        print(f"  {CIFAR10_CLASSES[i]:10s}: {logit.item():8.3f}")

    print("\nConverted to probabilities:")
    for i, p in enumerate(probs[0]):
        print(
            f"  {CIFAR10_CLASSES[i]:10s}: {p.item():8.3f} ({p.item()*100:.1f}%)")

    pred_idx = torch.argmax(raw_output, dim=1).item()
    print(f"\nPredicted class: {CIFAR10_CLASSES[pred_idx]}")
    print(f"Confidence: {probs[0][pred_idx].item():.3f}")


if __name__ == "__main__":
    try:
        print("ResNet-14 Inference Demo")
        print("=" * 40)

        if not os.path.exists("resnet14_cifar10.pth"):
            print("Error: Model file 'resnet14_cifar10.pth' not found!")
            print("Please run the training script first.")
            exit(1)

        test_on_cifar_samples()
        analyze_model_output()

        print("\n" + "=" * 50)
        print("Inference demo completed!")
        print("The model outputs 10 numbers for each image.")
        print("The highest number indicates the predicted class.")
        print("=" * 50)

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
