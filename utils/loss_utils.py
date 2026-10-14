from torch.nn.functional import binary_cross_entropy

class WBCE():
    def __init__(self, weight_bg):
        self.weight_bg = weight_bg

    def __call__(self, out, gt):

        mask_fg = gt > 0
        mask_bg = ~mask_fg

        out_fg = out[mask_fg]
        gt_fg = gt[mask_fg]
        out_bg = out[mask_bg]
        gt_bg = gt[mask_bg]

        if out_fg.nelement() <= 0:
            return self.weight_bg * binary_cross_entropy(out_bg, gt_bg)
        elif out_bg.nelement() <= 0:
            return binary_cross_entropy(out_fg, gt_fg)
        else:
            loss_fg = binary_cross_entropy(out_fg, gt_fg)
            loss_bg = binary_cross_entropy(out_bg, gt_bg)
            return loss_fg + self.weight_bg * loss_bg