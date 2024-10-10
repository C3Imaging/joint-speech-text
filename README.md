# Joint Speech-Text Embeddings for Multitask Speech Processing

*Michael Gian Gonzales<sup>1</sup>, Peter Corcoran<sup>1</sup>, Naomi Harte<sup>2</sup>, Michael Schukat<sup>1</sup>*

<sup>1</sup>University of Galway, <sup>2</sup>Trinity College Dublin


**Abstract:** Devices that use speech as the communication medium between human and computer have been emerging for the past few years. The technologies behind this interface are called Automatic Speech Recognition (ASR) and Text-to-Speech (TTS). The two are distinct fields in speech signal processing that have independently made great strides in recent years. This paper proposes an architecture that takes advantage of the two modalities present in ASR and TTS, speech and text, while simultaneously training three tasks, adding speaker recognition to the underlying ASR and TTS tasks. This architecture not only reduces the memory footprint required to run all tasks, but also has performance comparable to single-task models. The dataset used to train and evaluate the model is the CSTR VCTK Corpus. Results show a 97.64\% accuracy in the speaker recognition task, word and character error rates of 18.18\% and 7.95\% for the ASR task, a mel cepstral distortion of 4.31 and two predicted MOS of 2.98 and 3.28 for the TTS task. While voice conversion is not part of the training tasks, the architecture is capable of doing this and was evaluated to have 5.22, 2.98, and 2.73 for mel cepstral distortion and predicted MOS, respectively.

---

## Overview
[Model Inference]: https://github.com/C3Imaging/joint-speech-text/blob/main/images/Model_Inference.png
The proposed architecture is a model that can do three speech processing tasks: Speaker Recognition, Automatic Speech Recognition, and Text-to-Speech

![Model Inference]

---

## Usage

### Creating the environment
To install required packages and dependencies, poetry files are available. If poetry is unavailable, the requirements can be found inside **pyproject.toml** and be installed manually.

**Parallel Wavegan** should also be installed for the vocoder inference. A **kenlm** statistical language model (4-gram arpa) should also be present in the *langauge_model/* directory.

### Dataset Feature Extraction
In **modules/dataset.py**, the class named **FeatExtract_VCTK** extracts the following features:

- min-max normalized audio
- tokenized phonemes using g2p_en for TTS task
- tokenized capital letter targets for ASR task
- index of speaker identity for Speaker Recognition task
- mel-spectrogram target for TTS task

### Model Loading, Training, and Evaluation

The **modules** directory contains both the building blocks of a model and the model itself. The **jst\_\*** files contain the models:

- jst_model: the main joint speech-text model

The **scripts** directory contains both the training and evaluation scripts. The main script for training is the **train_all_ddp_gradaccum.py**. The script is outlined to do the following:

1. Setup DDP based on the number of GPUs available.
2. Setup logging utilities
3. Setup dataloaders
4. Setup and load models and optimizers either from scratch or checkpoint
5. In the training loop, inputs are forwarded to the model and losses are outputted.
   5.1 Accumulate gradients if batch/step number is not divisible by the set accumulation
   5.2 Update model parameters
6. Log the losses
7. Save the model when the epoch is a multiple of the set checkpoint number

---

## TTS Samples
Synthesized and target speech samples can be found in *tts_samples* directory. Prompts are as follows:

1. **17_0**: Try to save us
2. **117_0**: But he failed to win the committee's support
3. **217_0**: They bought the property four years ago
4. **317_0**: Now there's an even bigger target
5. **417_0**: That would be dangerous

---

## ASR Samples
Reference speech samples can be found in *asr_samples* directory. Predicted and target utterances are as follows:

Filename | Predicted | Target
--- | --- | --- |
**17_1** | THAT WAS THE EASY ELECTION | THAT WAS THE EASY ELECTION
**117_1** | HE ALSO LAUNCHED A NEW STRATEGY FOR THE AGENCY | HE ALSO LAUNCHED A NEW STRATEGY FOR THE AGENCY
**217_1** | IT'S VERY DIFFICULT TO FIND A BUYER | IT'S VERY DIFFICULT TO FIND A BUYER
**317_1** | INSTEAD SHE JUST WIN AWAY | INSTEAD SHE JUST WENT AWAY
**417_1** | YOU CAN EXPECT A HELLO D A FUSE | YOU CAN EXPECT A HELL OF A FUSS
