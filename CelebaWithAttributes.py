import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

class CelebAWithAttributes(Dataset):
    def __init__(self, img_dir, attr_csv, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images.
            attr_csv (str): Path to the CSV file with text attributes.
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.img_dir = img_dir
        self.transform = transform

        # Read the CSV file containing attributes.
        # The CSV is assumed to have a first column that is the image filename or ID.
        self.attr_df = pd.read_csv(attr_csv)
        
        # Determine which column holds the image file names.
        # Often for CelebA, the first column is something like 'image_id'
        if 'image_id' in self.attr_df.columns:
            self.image_column = 'image_id'
        else:
            self.image_column = self.attr_df.columns[0]
        
        # All other columns are considered attribute columns.
        self.attribute_columns = [col for col in self.attr_df.columns if col != self.image_column]

    def __len__(self):
        return len(self.attr_df)
    
    def __getitem__(self, idx):
        # Get the row associated with the sample
        row = self.attr_df.iloc[idx]
        img_name = row[self.image_column]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")
        
        if self.transform:
            image = self.transform(image)
        
        # Extract attributes from the row.
        # Here we assume attributes are represented as numerical flags (e.g., -1, 1).
        # For a text description, we list the attribute names that are positive (value == 1).
        attributes = row[self.attribute_columns].to_dict()
        positive_attrs = [attr for attr, val in attributes.items() if int(val) == 1]
        # You can customize the format of the text attribute description as needed.
        attr_text = ", ".join(positive_attrs)
        
        return image, attr_text