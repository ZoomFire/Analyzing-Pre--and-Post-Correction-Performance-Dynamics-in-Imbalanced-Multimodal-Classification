# import torch
# from src.phase2.dataset import get_dataloaders

# from .train import train_model
# from .evaluate import evaluate
# from .generate_results import generate_table, ensemble_predictions
# from .utils import evaluate_metrics


# def main():

#     # 🔥 Device setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n🔥 Using device: {device}")

#     # 🔥 DataLoader (ensure pin_memory for GPU speed)
#     train_loader, test_loader = get_dataloaders()

#     techniques = ["focal", "class_balanced", "mixup", "cutmix"]

#     results = []
#     models = []

#     for tech in techniques:

#         print(f"\n🚀 Running: {tech}")

#         # 🔥 Train (model already moved to GPU inside train_model)
#         model = train_model(train_loader, tech, device)
#         model.to(device)

#         models.append(model)

#         # 🔥 Evaluate (GPU optimized)
#         metrics = evaluate(model, test_loader, device)
#         metrics["Technique"] = tech

#         results.append(metrics)

#     # 🔥 ENSEMBLE
#     print("\n🔥 Running Ensemble")

#     preds = ensemble_predictions(models, test_loader, device)

#     # 🔥 Efficient ground truth collection
#     y_true = []
#     for _, labels in test_loader:
#         y_true.extend(labels.cpu().numpy())

#     ensemble_metrics = evaluate_metrics(y_true, preds)
#     ensemble_metrics["Technique"] = "Ensemble"

#     results.append(ensemble_metrics)

#     # 🔥 Save results
#     generate_table(results)


# if __name__ == "__main__":
#     main()

import torch
import os
from src.phase2.dataset import get_dataloaders

from .train import train_model
from .evaluate import evaluate
from .generate_results import generate_table, ensemble_predictions
from .utils import evaluate_metrics


def main():

    # 🔥 Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Using device: {device}")

    # 🔥 DataLoader
    train_loader, test_loader = get_dataloaders()

    techniques = ["focal", "class_balanced", "mixup", "cutmix"]

    results = []
    models = []

    for tech in techniques:

        print(f"\n🚀 Running: {tech}")

        model_path = f"model_{tech}.pth"

        # 🔥 TRAIN ONLY IF MODEL NOT EXISTS
        if not os.path.exists(model_path):
            print(f"⚡ Training {tech} model...")
            model = train_model(train_loader, tech, device)

            # save model
            torch.save(model.state_dict(), model_path)

        else:
            print(f"✅ Loading saved model: {model_path}")

            # ⚠️ IMPORTANT: same architecture load karo
            model = train_model(train_loader, tech, device)  # architecture milega
            model.load_state_dict(torch.load(model_path))

        model.to(device)
        model.eval()

        models.append(model)

        # 🔥 Evaluate
        metrics = evaluate(model, test_loader, device)
        metrics["Technique"] = tech

        results.append(metrics)

    # 🔥 ENSEMBLE
    print("\n🔥 Running Ensemble")

    preds = ensemble_predictions(models, test_loader, device)

    y_true = []
    for _, labels in test_loader:
        y_true.extend(labels.cpu().numpy())

    ensemble_metrics = evaluate_metrics(y_true, preds)
    ensemble_metrics["Technique"] = "Ensemble"

    results.append(ensemble_metrics)

    # 🔥 Save results (table + maybe CSV)
    generate_table(results)


if __name__ == "__main__":
    main()