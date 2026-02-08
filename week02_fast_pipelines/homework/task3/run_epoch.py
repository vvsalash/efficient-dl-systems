import typing as tp

import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import tqdm

import os
from profiler import Profile

from utils import Settings, Clothes, seed_everything
from vit import ViT


def get_vit_model() -> torch.nn.Module:
    model = ViT(
        depth=12,
        heads=4,
        image_size=224,
        patch_size=32,
        num_classes=20,
        channels=3,
    ).to(Settings.device)
    return model


def get_loaders() -> torch.utils.data.DataLoader:
    dataset.download_extract_dataset()
    train_transforms = dataset.get_train_transforms()
    val_transforms = dataset.get_val_transforms()

    frame = pd.read_csv("clothing-dataset/images.csv")
    img_dir = "clothing-dataset/images"

    paths = frame["image"].astype(str).apply(lambda x: os.path.join(img_dir, f"{x}.jpg"))
    exists = paths.apply(os.path.exists)
    frame = frame[exists].reset_index(drop=True)

    train_frame = frame.sample(frac=Settings.train_frac)
    val_frame = frame.drop(train_frame.index)

    train_data = dataset.ClothesDataset(
        "clothing-dataset/images", train_frame, transform=train_transforms
    )
    val_data = dataset.ClothesDataset(
        "clothing-dataset/images", val_frame, transform=val_transforms,
    )

    print(f"Train Data: {len(train_data)}")
    print(f"Val Data: {len(val_data)}")

    train_loader = DataLoader(dataset=train_data, batch_size=Settings.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=Settings.batch_size, shuffle=False)

    return train_loader, val_loader


def build_layer_table(events: list[dict]) -> pd.DataFrame:
    rows = []
    for event in events:
        args = event.get("args", {})
        rows.append(
            {
                "name": event.get("name", ""),
                "kind": args.get("kind", ""),
                "phase": args.get("phase", ""),
                "step": args.get("step", -1),
                "dur_us": float(event.get("dur", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["phase"] == "active"].copy()

    agg = (
        df.groupby(["name", "kind"], as_index=False)
        .agg(
            total_us=("dur_us", "sum"),
            mean_us=("dur_us", "mean"),
            p50_us=("dur_us", "median"),
            max_us=("dur_us", "max"),
            count=("dur_us", "count"),
        )
        .sort_values("total_us", ascending=False)
    )
    return agg


def run_epoch(model, train_loader, val_loader, criterion, optimizer) -> tp.Tuple[float, float]:
    epoch_loss, epoch_accuracy = 0, 0
    val_loss, val_accuracy = 0, 0
    model.train()
    for data, label in tqdm(train_loader, desc="Train"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc.item() / len(train_loader)
        epoch_loss += loss.item() / len(train_loader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    for data, label in tqdm(val_loader, desc="Val"):
        data = data.to(Settings.device)
        label = label.to(Settings.device)
        output = model(data)
        loss = criterion(output, label)
        acc = (output.argmax(dim=1) == label).float().mean()
        val_accuracy += acc.item() / len(train_loader)
        val_loss += loss.item() / len(train_loader)

    return epoch_loss, epoch_accuracy, val_loss, val_accuracy


def profile_train_steps(
    model,
    train_loader,
    criterion,
    optimizer,
    profile_steps: int = 10,
    schedule: dict | None = None,
    trace_path: str = "trace.json",
    table_path: str = "layer_times.csv",
) -> None:
    schedule = schedule or {"wait": 1, "warmup": 1, "active": profile_steps, "repeat": 1}

    model.train()
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(table_path) or ".", exist_ok=True)

    with Profile(model, name="vit", schedule=schedule) as prof:
        for i, (data, label) in enumerate(tqdm(train_loader, desc="ProfileTrain")):
            prof.step()
            data = data.to(Settings.device)
            label = label.to(Settings.device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if i + 1 >= (schedule["wait"] + schedule["warmup"] + schedule["active"]):
                break

    prof.to_perfetto(trace_path)

    print("spans:", len(prof._spans))
    print("events:", len(prof.events))

    df = build_layer_table(prof.events)
    df.to_csv(table_path, index=False)

    if not df.empty:
        print(df.head(15).to_string(index=False))


def main():
    print("cuda available:", torch.cuda.is_available())
    print("device:", Settings.device)
    seed_everything()
    model = get_vit_model()
    train_loader, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Settings.lr)

    # run_epoch(model, train_loader, val_loader, criterion, optimizer)

    profile_train_steps(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        profile_steps=10,
        schedule={"wait": 1, "warmup": 1, "active": 10, "repeat": 1},
        trace_path="trace.json",
        table_path="layer_times.csv",
    )



if __name__ == "__main__":
    main()
