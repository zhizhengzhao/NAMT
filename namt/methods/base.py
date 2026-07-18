class Method:
    name = "base"
    tier = "A"

    def calibrate(self, blank_hits, layer_z, sigma_pos):
        return {}

    def reconstruct(self, hits, layer_z, sigma_pos, cal, vox, **kw):
        raise NotImplementedError
