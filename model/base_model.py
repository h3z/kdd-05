from train import losses, optimizers, schedulers


class BaseModelApp:
    def __init__(self, *models) -> None:
        self.model = models[0]
        self.opt = optimizers.get(self.model)
        self.loss = losses.get()
        self.scheduler = None

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()
        if self.scheduler:
            self.scheduler.step()

    def forward(self, batch_x, batch_y, is_training=False):
        return self.model(batch_x)

    def checkpoint(self):
        return self.model.state_dict()

    def load_checkpoint(self, state_dict):
        self.model.load_state_dict(state_dict)

    def init_scheduler(self, batch_count):
        self.scheduler = schedulers.get(self.opt, batch_count)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def criterion(self, pred_y, batch_y):
        return self.loss(pred_y, batch_y)
