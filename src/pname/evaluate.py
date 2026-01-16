import torch
import typer
from loguru import logger
from pname.data import corrupt_mnist
from pname.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    logger.info("Evaluating like my life depended on it")
    logger.info(f"Loading model from: {model_checkpoint}")

    model = MyAwesomeModel().to(DEVICE)
    model = model.float()
    # Load model checkpoint, mapping to CPU first to handle cross-platform compatibility
    checkpoint = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    logger.info("Model loaded successfully")

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)
    logger.info(f"Test dataset size: {len(test_set)}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(test_dataloader):
            img = img.float().to(DEVICE)
            target = target.long().to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Processed {batch_idx + 1}/{len(test_dataloader)} batches")

    test_accuracy = correct / total
    logger.info(f"Test accuracy: {test_accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    typer.run(evaluate)
