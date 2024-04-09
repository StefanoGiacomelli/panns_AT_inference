import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_AT_inference
from panns_AT_inference import AudioTagging, labels


def print_audio_tagging_result(clipwise_output):
    """Visualization of audio tagging result.

    Args:
      clipwise_output: (classes_num,)
    """
    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))


if __name__ == '__main__':
    """Example of using panns_inferece for audio tagging.
    """
    device = 'cpu' # 'cuda' | 'cpu'
    audio_path = 'resources/R9_ZSCveAHg_7s.wav'
    (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]  # (batch_size, segment_samples)

    print('------ Audio tagging ------')
    at = AudioTagging(checkpoint_path=None, device=device)
    (clipwise_output, embedding) = at.inference(audio)
    """clipwise_output: (batch_size, classes_num), embedding: (batch_size, embedding_size)"""

    print_audio_tagging_result(clipwise_output[0])
