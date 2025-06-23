from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from Dataset.Dataset_EM import MyDataset_EM
from pytorch_lightning.loggers import CSVLogger

# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 2000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


input_folder="/home/adrienb/Documents/Data/dataset_2_ControlNet_256_256/patches"
# Misc
dataset = MyDataset_EM(input_folder=input_folder)
print("dataset loaded correctly.")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32,max_epochs=5,logger=CSVLogger("logs", name="controlnet_run"), callbacks=[logger])

print("starting training...")
# Train!
trainer.fit(model, dataloader)
