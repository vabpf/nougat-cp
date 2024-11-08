import os
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM

class CustomDataset(Dataset):
    def __init__(self, image_dir: str, text_dir: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.image_dir = Path(image_dir)
        self.text_dir = Path(text_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_paths = list(self.image_dir.glob("*.png"))
        self.text_paths = list(self.text_dir.glob("*.txt"))
        assert len(self.image_paths) == len(self.text_paths), "Mismatch between number of images and text files"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.prepare_image(image)

        # Load ground truth text
        text_path = self.text_paths[idx]
        with open(text_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        # Tokenize ground truth text
        tokenizer_out = self.tokenizer(
            ground_truth,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenizer_out["input_ids"].squeeze(0)
        attention_mask = tokenizer_out["attention_mask"].squeeze(0)

        return image_tensor, input_ids, attention_mask

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        # Implement any image preprocessing required by the model here
        # For example, resizing, normalization, etc.
        image = image.resize((224, 224))  # Example resize
        image_tensor = torch.tensor(image).permute(2, 0, 1)  # Convert to tensor and rearrange dimensions
        return image_tensor

# Example usage
model_dir = r"D:\Models\torch\hub\nougat-0.1.0-small"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

dataset = CustomDataset(image_dir=r"D:\Data\vietnamese_ocr_data\InkData_line_processed", text_dir=r"D:\Data\vietnamese_ocr_data\InkData_line_processed", tokenizer=tokenizer, max_length=512)

# Now you can use this dataset with a DataLoader for training
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)