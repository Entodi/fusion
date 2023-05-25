from fusion.criterion.loss.dim import BaseDim


CR_MODE = "CR"


class CrDim(BaseDim):
    _name = CR_MODE

    def __call__(self, reps, convs):
        total_loss = None
        raw_losses = {}
        for source_id, rep in reps.items():
            dim_conv_latent = list(rep.keys())[-1]
            assert int(dim_conv_latent.split('_')[0]) == 1
            for dim_conv, conv in convs[source_id].items():
                assert dim_conv_latent in rep.keys()
                name = self._name_it(source_id, dim_conv)
                loss, penalty = self._estimator(conv, rep[dim_conv_latent])
                total_loss, raw_losses = self._update_loss(
                    name, total_loss, raw_losses, loss, penalty
                )
        return total_loss, raw_losses

    def _name_it(self, source_id, dim_conv):
        return f"{self._name}{dim_conv}_{source_id}"
