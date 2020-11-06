

class CosmosCutout(Cutout):
    @staticmethod
    def psf_function(r):
        # usually r = 0.3 * pix
        return galsim.Moffat(2, r)

    def get_psf(self, psf_stamp_size):
        assert psf_stamp_size % 2 == 1
        psf_int = self.psf_function(self.psf_args).withFlux(1.0)

        # Draw PSF
        psf = psf_int.drawImage(
            nx=self.psf_size,
            ny=self.psf_size,
            method="real_space",
            use_true_center=True,
            scale=self.pix,
        ).array

        # Make sure PSF vanishes on the edges of a patch that
        # has the shape of the initial npsf
        psf = psf - psf[0, int(self.psf_size / 2)] * 2
        psf[psf < 0] = 0
        psf = psf / np.sum(psf)

        return psf
