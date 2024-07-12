import random

import torchaudio
import torchvision.transforms as transforms


def compute_mel_spectrogram(audio_path, mel_spectrogram, amplitude_to_db, n_mels=256):
    """
    Compute a Mel spectrogram for an audio file.

    Args:
    audio_path (str): Path to the audio file.
    n_fft (int): Number of FFT components.
    hop_length (int): Number of samples between successive frames.
    n_mels (int): Number of Mel bands to generate.
    sr (int): Sampling rate of the audio file.

    Returns:
    S_DB (np.array): Log-scaled Mel spectrogram.
    """
    # Load the audio file
    waveform, _ = torchaudio.load(audio_path)
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[mel_spec_db.mean()], std=[mel_spec_db.std() + 1e-8]
            ),
            transforms.Resize((n_mels, 200)),
        ]
    )

    mel_spec_transformed = transform(mel_spec_db)

    return mel_spec_transformed


def set_random_rows_to_zero(image_tensor, max_consecutive_rows):
    height, _ = image_tensor.shape[0], image_tensor.shape[1]
    # Handle case where the height is less than the max consecutive rows
    if height <= max_consecutive_rows:
        num_rows = (
            height  # Use the entire height if it's less than or equal to max rows
        )
    else:
        num_rows = random.randint(0, max_consecutive_rows)

    if num_rows > 0:  # Only modify the tensor if num_rows is greater than zero
        start_row = random.randint(0, height - num_rows)
        image_tensor[start_row : start_row + num_rows, :] = 0

    return image_tensor


def set_random_columns_to_zero(image_tensor, max_consecutive_columns):
    _, width = image_tensor.shape[0], image_tensor.shape[1]
    # Handle case where the width is less than the max consecutive columns
    if width <= max_consecutive_columns:
        num_columns = (
            width  # Use the entire width if it's less than or equal to max columns
        )
    else:
        num_columns = random.randint(0, max_consecutive_columns)

    if num_columns > 0:  # Only modify the tensor if num_columns is greater than zero
        start_column = random.randint(0, width - num_columns)
        image_tensor[:, start_column : start_column + num_columns] = 0

    return image_tensor


def apply_spectral_augment(image, num):
    if num == 1:
        aug_image = set_random_rows_to_zero(
            image, 20
        )  # 20 rows in the frequency domain
        return aug_image
    elif num == 2:
        aug_image = set_random_columns_to_zero(
            image, 30
        )  # 30 columns in the time domain
        return aug_image
    else:
        aug_image1 = set_random_rows_to_zero(image, 20)
        aug_image = set_random_columns_to_zero(aug_image1, 30)
        return aug_image


def compute_mel_spectrogram_aug(
    audio_path, mel_spectrogram, amplitude_to_db, n_mels=256
):
    """
    Compute a Mel spectrogram for an audio file.

    Args:
    audio_path (str): Path to the audio file.
    n_fft (int): Number of FFT components.
    hop_length (int): Number of samples between successive frames.
    n_mels (int): Number of Mel bands to generate.
    sr (int): Sampling rate of the audio file.

    Returns:
    S_DB (np.array): Log-scaled Mel spectrogram.
    """
    # Load the audio file

    waveform, _ = torchaudio.load(audio_path)
    waveform = waveform
    mel_spec = mel_spectrogram(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[mel_spec_db.mean()], std=[mel_spec_db.std() + 1e-8]
            ),  # Clamp standard deviation
            transforms.Resize(
                (n_mels, 200)
            ),  # Required second dimension is necessarily 200
        ]
    )

    mel_spec_transformed = transform(mel_spec_db)

    num = random.randint(1, 3)
    aug_mel = apply_spectral_augment(mel_spec_transformed, num)
    return aug_mel
