from fusion.criterion.loss.dim import BaseDim


RR_MODE = "RR"


class RrDim(BaseDim):
    _name = RR_MODE

    def __call__(self, reps, convs):
        total_loss = None
        raw_losses = {}
        for rep_source_id_one, rep_one in reps.items():
            dim_conv_latent = list(rep_one.keys())[-1]
            assert int(dim_conv_latent.split('_')[0]) == 1
            for rep_source_id_two, rep_two in reps.items():
                if rep_source_id_one != rep_source_id_two:
                    #print (rep_two.keys(), rep_one.keys())
                    #print (dim_conv_latent)
                    #assert dim_conv_latent in rep_two.keys()
                    #assert dim_conv_latent in rep_one.keys()
                    name = self._name_it(
                        rep_source_id_one, rep_source_id_two, dim_conv_latent
                    )
                    loss, penalty = self._estimator(
                        rep_one[dim_conv_latent], rep_two[dim_conv_latent]
                    )
                    total_loss, raw_losses = self._update_loss(
                        name, total_loss, raw_losses, loss, penalty
                    )
        return total_loss, raw_losses

    def _name_it(self, rep_source_id, conv_source_id, dim_conv):
        return f"{self._name}{dim_conv}_" f"{rep_source_id}_{conv_source_id}"
