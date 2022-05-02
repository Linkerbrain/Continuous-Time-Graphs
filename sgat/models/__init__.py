import pytorch_lightning as pl

class SgatModule(pl.LightningModule):
    def __init__(self, graph, params, train_dataloader_gen, val_dataloader_gen):
        self.graph = graph
        self.params = params
        self.train_dataloader_gen = train_dataloader_gen
        self.val_dataloader_gen = val_dataloader_gen

    def train_dataloader(self):
        self.train_dataloader_gen(self.current_epoch)

    def val_dataloader(self):
        self.val_dataloader_gen(self.current_epoch)
