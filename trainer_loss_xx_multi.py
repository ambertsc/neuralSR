"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, best=None, device='gpu', uniqueID='0000'):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.loss_size = 0

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if device == 'gpu' and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print('We are using the gpu now! device={}'.format(self.device))

        self.uniqueID = str(uniqueID)
        self.best_loss = best
        self.hold_losses = []
        self.train_losses = []
        self.val_losses = []


    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split):
            self.hold_losses = []
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            logits_printed = True
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, p, v) in pbar:

                # place data on the correct device
                x = x.to(self.device) # input equation
                y = y.to(self.device) # output equation
                p = p.to(self.device) # points
                v = v.to(self.device) # number of variables

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y, p, v, tokenizer=self.train_dataset.itos)
                    mean_loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(mean_loss.item())
                    if logits_printed == False:
                        #
                        # for i in range(logits.size()[0]):
                        #     current_eqn = logits[i]
                        #     print("Current Equation")
                        #     print(type(current_eqn))
                        #     print(current_eqn.size())
                        #     print("---------------------------------------------------")

                        logits_printed = True




                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    mean_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {mean_loss.item():.5f}. lr {lr:e}")
                    print(type(loss))
                    print(loss)
                    if self.loss_size == 0:
                        print("The loss size after changes is ", loss.size())
                        self.loss_size = 1
                    try:
                        self.hold_losses.extend(loss.tolist())
                    except:
                        self.hold_losses.append(loss.item())

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        self.best_loss = float('inf') if self.best_loss is None else self.best_loss
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            self.train_losses.append(np.mean(self.hold_losses))
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                self.val_losses.append(test_loss)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < self.best_loss
            if self.config.ckpt_path is not None and good_model:
                self.best_loss = test_loss
                self.save_checkpoint()

        val_array = np.asarray(self.val_losses)
        train_array = np.asarray(self.train_losses)
        np.save('withpadding/val_loss_xx'+str(self.uniqueID), val_array)
        np.save('withpadding/train_loss_xx'+str(self.uniqueID), train_array)
