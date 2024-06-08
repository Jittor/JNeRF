
class BaseTrainer():

    def __init__(self, cfg, args):
        pass

    def update(self, data, *args, **kwargs):
        raise NotImplementedError("Trainer [update] not implemented.")

    def epoch_end(self, epoch, writer=None, **kwargs):
        # Signal now that the epoch ends....
        pass

    def multi_gpu_wrapper(self, wrapper):
        raise NotImplementedError("Trainer [multi_gpu_wrapper] not implemented.")

    def log_train(self, train_info, train_data,
                  writer=None, step=None, epoch=None, visualize=False,
                  **kwargs):
        raise NotImplementedError("Trainer [log_train] not implemented.")

    def validate(self, test_loader, epoch, *args, **kwargs):
        raise NotImplementedError("Trainer [validate] not implemented.")

    def log_val(self, val_info, writer=None, step=None, epoch=None, **kwargs):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    writer.add_scalar(k, v, step)
                else:
                    writer.add_scalar(k, v, epoch)

    def save(self, epoch=None, step=None, appendix=None, **kwargs):
        raise NotImplementedError("Trainer [save] not implemented.")

    def resume(self, path, strict=True, **kwargs):
        raise NotImplementedError("Trainer [resume] not implemented.")
