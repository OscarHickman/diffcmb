"""Minimal runner for the refactored package."""
from diffcmb.model import CosmologyAdvancedSampling


def main():
    model = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)
    print("Constructed CosmologyAdvancedSampling with:")
    print(f" lmax={model.lmax}, NSIDE={model.NSIDE}, NPIX={model.NPIX}")


if __name__ == "__main__":
    main()
