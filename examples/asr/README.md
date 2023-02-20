## Running Whisper on edge with BentoML and whisper.cpp

This example demonstrates how to run Whisper on edge with BentoML and
whisper.cpp using a custom CPP Runner.

## Instruction

Install required dependencies:

```bash
pip install -r requirements.txt
```

To load a pretrained model, use `Whisper.from_pretrained()`:

```python
from whispercpp import Whisper

model = Whisper.from_pretrained("tiny.en")

# preprocess audio file and transcribe. You can use any preprocessing library you wish.
# the example uses librosa for convenience.
import librosa
import numpy as np
audio, _ = librosa.load("/path/to/audio.wav")
model.transcribe(audio.astype(np.float32))
```

### Building bento

To package the bento, use `build_bento.py`:

```python
python build_bento.py
```

To override existing bento, pass in `--overrride`:

```python
python build_bento.py --override
```

### Containerize bento

To containerize the bento, run `bentoml containerize`:

```python
bentoml containerize whispercpp_asr
```
