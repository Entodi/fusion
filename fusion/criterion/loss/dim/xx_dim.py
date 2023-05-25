from fusion.criterion.loss.dim import BaseDim


XX_MODE = "XX"


class XxDim(BaseDim):
    _name = XX_MODE

    def __call__(self, reps, convs):
        total_loss = None
        raw_losses = {}
        for rep_source_id, rep in reps.items():
            dim_conv_latent = list(rep.keys())[-1]
            assert int(dim_conv_latent.split('_')[0]) == 1
            for conv_source_id, conv in convs.items():
                if rep_source_id != conv_source_id:
                    for dim_conv, conv_latent in conv.items():
                        #assert dim_conv_latent in rep.keys()
                        name = self._name_it(rep_source_id, conv_source_id, dim_conv)
                        loss, penalty = self._estimator(
                            conv_latent, rep[dim_conv_latent]
                        )
                        loss = self._weight * loss
                        total_loss, raw_losses = self._update_loss(
                            name, total_loss, raw_losses, loss, penalty
                        )
        return total_loss, raw_losses

    def _name_it(self, rep_source_id, conv_source_id, dim_conv):
        return f"{self._name}{dim_conv}_" f"{rep_source_id}_{conv_source_id}"
