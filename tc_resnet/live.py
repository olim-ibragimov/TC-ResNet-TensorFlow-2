from os import listdir
from os.path import isdir, join

import hydra
import sounddevice as sd
import soundfile as sf
import numpy as np
# from tensorflow.keras.models import load_model

from data_preprocessor import DataPreprocessor
from tc_resnet import PROJECT_PATH
from model import get_tc_resnet_8


@hydra.main(config_path='configs', config_name='live')
def live(params):
    sr = params['sampling_rate']
    audio_length = params['audio_length']
    blocksize = params['blocksize']
    sd.default.samplerate = sr
    sd.default.channels = params['channels']
    sd.default.blocksize = blocksize
    sd.default.latency = params['latency']

    weights_path = PROJECT_PATH / params['weights_file']

    classes = list(params['all_classes'])
    num_classes = len(classes)

    # model = get_tc_resnet_14((321, 40), num_classes, 1.5)
    model = get_tc_resnet_8((321, 40), num_classes, 1.5)
    model.load_weights(weights_path)
    model.summary()

    recent_signal = []
    recording_id = 0

    try:
        while True:
            input("Press Enter to start recording:")
            stream = sd.InputStream()
            stream.start()
            print("Say the word:")
            while True:
                data, overflowed = stream.read(blocksize)
                data = data.flatten()
                recent_signal.extend(data.tolist())
                if len(recent_signal) >= sr * audio_length:
                    recent_signal = recent_signal[:sr * audio_length]
                    break
            stream.close()
            rec_path = PROJECT_PATH / f'recording_{recording_id}.wav'
            sf.write(rec_path, np.array(recent_signal), sr)
            recording_id += 1
            print("Recording finished! Result is:")
            mfcc = DataPreprocessor.get_mfcc(np.asarray(recent_signal), sr)
            y_pred = model.predict(np.array([mfcc]))[0]
            result_id = int(np.argmax(y_pred))
            result_prob = y_pred[result_id]
            print("result id: " + str(result_id) + " " + classes[result_id] + " " + str(result_prob))
            recent_signal = []
    except KeyboardInterrupt:
        print('Record finished!')


if __name__ == '__main__':
    live()
