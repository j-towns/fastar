import fnmatch
import os
import random
import re

import librosa
import numpy as np

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate=None):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        if sample_rate is None:
            audio, _ = librosa.load(filename, sr=None, mono=True)  # keep the true sampling rate
        else:
            audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rms(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def audio_loader(audio_dir, receptive_field,
                 sample_size, silence_threshold,
                 batch_size):
    """This returns a generator over batches.
    Note that batches are generated on the fly,
    so it will be slower than loading all data once,
    but will have less memory footprint"""
    files = find_files(audio_dir)
    if not files:
        raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def generator():
        data = []
        iterator = load_generic_audio(audio_dir)
        stop = False
        for audio, filename, _ in iterator:
            if stop:
                break
            audio = trim_silence(audio[:, 0], silence_threshold)
            audio = audio.reshape(-1, 1)
            if audio.size == 0:
                print("Warning: {} was ignored as it contains only "
                      "silence. Consider decreasing trim_silence "
                      "threshold, or adjust volume of the audio."
                      .format(filename))

            # pad with zeros at front
            audio = np.pad(audio, [[receptive_field, 0], [0, 0]], 'constant')

            # now trim down to pieces of sample_size
            while len(audio) > receptive_field:
                piece = audio[:(receptive_field + sample_size), :]
                # make sure our data is of the right size so it can be batched
                if len(piece) == sample_size + receptive_field:
                    data.append(piece)
                if len(data) > 0 and len(data) % batch_size == 0:
                    yield np.stack(data).astype('float32')
                    data = []
                audio = audio[sample_size:, :]
    return generator
