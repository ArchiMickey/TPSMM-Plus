import lightning.pytorch as pl
from torch.utils.data import DataLoader
from tpsmm_plus.modules.data.util import FramesDataset, DatasetRepeater


class FrameDataModule(pl.LightningDataModule):
    def __init__(self, ds_params, num_repeats, batch_size=1, num_workers=1):
        super().__init__()
        self.ds_params = ds_params
        self.num_repeats = num_repeats
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.ds = FramesDataset(**self.ds_params)

    def train_dataloader(self):
        ds = DatasetRepeater(self.ds, self.num_repeats)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=self.num_workers)