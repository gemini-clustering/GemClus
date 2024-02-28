from ._fdivergences import MI, KLGEMINI, TVGEMINI, HellingerGEMINI
from ._geomdistances import MMDGEMINI, WassersteinGEMINI


def _str_to_gemini(gemini_str):
    if gemini_str not in AVAILABLE_GEMINIS:
        raise ValueError(f"Unknown GEMINI: {gemini_str}. Please choose among {AVAILABLE_GEMINIS}.")
    if gemini_str == "mmd_ova":
        return MMDGEMINI()
    elif gemini_str == "mmd_ovo":
        return MMDGEMINI(ovo=True)
    elif gemini_str == "wasserstein_ova":
        return WassersteinGEMINI()
    elif gemini_str == "wasserstein_ovo":
        return WassersteinGEMINI(ovo=True)
    elif gemini_str == "kl_ova" or gemini_str == "mi":
        return MI()
    elif gemini_str == "kl_ovo":
        return KLGEMINI(ovo=True)
    elif gemini_str == "tv_ova":
        return TVGEMINI()
    elif gemini_str == "tv_ovo":
        return TVGEMINI(ovo=True)
    elif gemini_str == "hellinger_ova":
        return HellingerGEMINI()
    elif gemini_str == "hellinger_ovo":
        return HellingerGEMINI(ovo=True)


AVAILABLE_GEMINIS = ["mmd_ova", "mmd_ovo", "wasserstein_ova", "wasserstein_ovo", "kl_ova", "kl_ovo", "mi", "tv_ova",
                     "tv_ovo", "hellinger_ova",  "hellinger_ovo"]
