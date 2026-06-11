import pandas as pd
import torch


def ensemble_predictions(models, dataloader, device):
    all_preds = []

    for model in models:
        model.to(device)
        model.eval()
        preds = []

        with torch.no_grad(), torch.cuda.amp.autocast():
            for images, _ in dataloader:
                images = images.to(device, non_blocking=True)

                outputs = model(images)
                preds.append(outputs.detach())

        all_preds.append(torch.cat(preds, dim=0))

    avg_preds = torch.mean(torch.stack(all_preds, dim=0), dim=0)
    return torch.argmax(avg_preds, dim=1).cpu().numpy()


def generate_table(results):
    df = pd.DataFrame(results)

    print("\n🔥 FINAL RESULT TABLE 🔥\n")
    print(df)

    df.to_csv("phase3_results.csv", index=False)