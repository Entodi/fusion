from .catalyst_runner import CatalystRunner


class OasisRunner(CatalystRunner):
    def _unpack_batch(self, batch):
        x = {
            k.split('_')[-1]: v['data'] for k, v in batch.items() if k.startswith('source')
        }
        y = batch["label"]
        return x, y
