# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import logging

import numpy as np
import pytest

from art.config import ART_NUMPY_DTYPE
from tests.utils import ARTTestException
from icefall.utils import AttributeDict
from pathlib import Path

logger = logging.getLogger(__name__)


@pytest.mark.skip_module("icefall")
@pytest.mark.skip_framework("tensorflow", "tensorflow2v1", "keras", "kerastf", "mxnet", "non_dl_frameworks")
@pytest.mark.parametrize("device_type", ["cpu"])
def test_pytorch_icefall(art_warning, expected_values, device_type):
    import torch

    from art.estimators.speech_recognition.pytorch_icefall import PyTorchIcefall
    from transducer.decode import get_id2word

    try:
        # construct icefall args
        params = get_params()
        Path(params['exp_dir']).mkdir(parents=True, exist_ok=True)
        Path(params['lang_dir']).mkdir(parents=True, exist_ok=True)
        

        # load_model_ensemble
        transducer_model = get_transducer_model(params)
        transducer_model.device = 'cpu'
        word2ids = get_word2id(params)
        get_id2word = get_id2word(params)
        model_ensemble = {
            'model': transducer_model,
            'word2ids': word2ids,
            'get_id2word': get_id2word,
            'params': params
        }

        # load model checkpoint
        if params.avg == 1:
            from icefall.checkpoint import load_checkpoint
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model_ensemble['model'])
        else:
            from icefall.checkpoint import average_checkpoints

            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if start >= 0:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model_ensemble['model'].load_state_dict(average_checkpoints(filenames))

        # Initialize a speech recognizer
        speech_recognizer = PyTorchIcefall(model_ensemble=model_ensemble)

        # Load data for testing
        expected_data = expected_values()

        x1 = expected_data["x1"]
        x2 = expected_data["x2"]
        x3 = expected_data["x3"]

        # Create signal data
        xs = [x1, x2, x3]

        # Create labels
        y = np.array([['<s>', 'activate', 'lights', 'none', '</s>'], ['<s>', 'increase', 'volume', 'none', '</s>'], ['<s>', 'deactivate', 'lights', 'washroom', '</s>']])

        # Test transcription outputs
        hyps = []
        hyps.append(speech_recognizer.predict(np.expand_dims(np.array(xs[0]), 0)))
        hyps.append(speech_recognizer.predict(np.expand_dims(np.array(xs[1]), 0)))
        hyps.append(speech_recognizer.predict(np.expand_dims(np.array(xs[2]), 0)))
        print(hyps)

    except ARTTestException as e:
        art_warning(e)


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    is saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - lr: It specifies the initial learning rate

        - feature_dim: The model input dim. It has to match the one used
                    in computing features.

        - weight_decay:  The weight_decay for the optimizer.

        - subsampling_factor:  The subsampling factor for the model.

        - start_epoch:  If it is not zero, load checkpoint `start_epoch-1`
                        and continue training from that checkpoint.

        - best_train_loss: Best training loss so far. It is used to select
                        the model that has the lowest training loss. It is
                        updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                        the model that has the lowest validation loss. It is
                        updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                        contains number of batches trained so far across
                        epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - valid_interval:  Run validation if batch_idx % valid_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0


    """
    from pathlib import Path

    params = AttributeDict(
        {
            "lr": 1e-3,
            "feature_dim": 23,
            "weight_decay": 1e-6,
            "start_epoch": 0,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 100,
            "reset_interval": 20,
            "valid_interval": 300,
            "exp_dir": Path("."),
            "lang_dir": Path("frames"),
            # encoder/decoder params
            "vocab_size": 3,  # blank, yes, no
            "blank_id": 0,
            "embedding_dim": 32,
            "hidden_dim": 16,
            "num_decoder_layers": 4,
            "epoch": 6,
            "avg": 1,
        }
    )

    vocab_size = 1
    try:
        with open(Path(params.lang_dir) / "lexicon_disambig.txt") as lexicon_file:
            for line in lexicon_file:
                if len(line.strip()) > 0:  # and '<UNK>' not in line and '<s>' not in line and '</s>' not in line:
                    vocab_size += 1
    except:
        pass
    params.vocab_size = vocab_size

    return params

def get_transducer_model(params: AttributeDict):
    from transducer.decoder import Decoder
    from transducer.conformer import Conformer
    from transducer.joiner import Joiner
    from transducer.model import Transducer

    encoder = Conformer(
        num_features=params.feature_dim,
        output_dim=params.hidden_dim,
    )
    decoder = Decoder(
        vocab_size=params.vocab_size,
        embedding_dim=params.embedding_dim,
        blank_id=params.blank_id,
        num_layers=params.num_decoder_layers,
        hidden_dim=params.hidden_dim,
        embedding_dropout=0.4,
        rnn_dropout=0.4,
    )
    joiner = Joiner(input_dim=params.hidden_dim, output_dim=params.vocab_size)
    transducer = Transducer(encoder=encoder, decoder=decoder, joiner=joiner)

    return transducer


def get_word2id(params):
    from pathlib import Path

    word2id = {}

    # 0 is blank
    id = 1
    try:
        with open(Path(params.lang_dir) / "lexicon_disambig.txt") as lexicon_file:
            for line in lexicon_file:
                if len(line.strip()) > 0:
                    word2id[line.split()[0]] = id
                    id += 1
    except:
        pass

    return word2id