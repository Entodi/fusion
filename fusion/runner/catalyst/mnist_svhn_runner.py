from .catalyst_runner import CatalystRunner


class MnistSvhnRunner(CatalystRunner):
    def _unpack_batch(self, batch):
        if len(batch) == 2:
            y = batch[0][1]
            x = {
                "0": batch[0][0]["0"],
                "1": batch[1][0]["1"]
            }
        else:
            x, y = batch[0][0], batch[0][1]
        return x, y
