import numpy as np


def combine_de_psd(de_features, psd_features):
    """
    Combine Differential Entropy (DE) and PSD features.

    Parameters
    ----------
    de_features : np.ndarray
        Shape: (segments, channels, 5)

    psd_features : np.ndarray
        Shape: (segments, channels, 5)

    Returns
    -------
    combined_features : np.ndarray
        Shape: (segments, channels, 10)
    """

    de_features = np.asarray(
        de_features,
        dtype=np.float32,
    )

    psd_features = np.asarray(
        psd_features,
        dtype=np.float32,
    )

    if de_features.ndim != 3:
        raise ValueError(
            "DE features must have shape "
            "(segments, channels, features)."
        )

    if psd_features.ndim != 3:
        raise ValueError(
            "PSD features must have shape "
            "(segments, channels, features)."
        )

    if de_features.shape[:2] != psd_features.shape[:2]:
        raise ValueError(
            "DE and PSD must have the same "
            "number of segments and channels."
        )

    combined_features = np.concatenate(
        [de_features, psd_features],
        axis=-1,
    )

    return combined_features


if __name__ == "__main__":
    print("Feature fusion module created successfully.")