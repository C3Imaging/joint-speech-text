# Joint Speech-Text Embeddings for Multitask Speech Processing

*Michael Gian Gonzales<sup>1</sup>, Peter Corcoran<sup>1</sup>, Naomi Harte<sup>2</sup>, Michael Schukat<sup>1</sup>*

<sup>1</sup>University of Galway, <sup>2</sup>Trinity College Dublin


**Abstract:** Devices that use speech as the communication medium between human and computer have been emerging for the past few years. The technologies behind this interface are called Automatic Speech Recognition (ASR) and Text-to-Speech (TTS). The two are distinct fields in speech signal processing that have independently made great strides in recent years. This paper proposes an architecture that takes advantage of the two modalities present in ASR and TTS, speech and text, while simultaneously training three tasks, adding speaker recognition to the underlying ASR and TTS tasks. This architecture not only reduces the memory footprint required to run all tasks, but also has performance comparable to single-task models. The dataset used to train and evaluate the model is the CSTR VCTK Corpus. Results show a 97.64\% accuracy in the speaker recognition task, word and character error rates of 18.18\% and 7.95\% for the ASR task, a mel cepstral distortion of 4.31 and two predicted MOS of 2.98 and 3.28 for the TTS task. While voice conversion is not part of the training tasks, the architecture is capable of doing this and was evaluated to have 5.22, 2.98, and 2.73 for mel cepstral distortion and predicted MOS, respectively.

*Full code will be released once the article has been published*

## Overview
[Model Inference]: https://github.com/C3Imaging/joint-speech-text/images/Model_Inference.png
The proposed architecture is a model that can do three speech processing tasks: Speaker Recognition, Automatic Speech Recognition, and Text-to-Speech

![Model Inference]

## TTS Samples
Synthesized and target speech samples can be found in *tts_samples* directory. Prompts are as follows:

1. **17_0**: Try to save us
2. **117_0**: But he failed to win the committee's support
3. **217_0**: They bought the property four years ago
4. **317_0**: Now there's an even bigger target
5. **417_0**: That would be dangerous

## ASR Samples
Reference speech samples can be found in *asr_samples* directory. Predicted and target utterances are as follows:

Filename | Predicted | Target
--- | --- | --- |
**17_1** | THAT WAS THE EASY ELECTION | THAT WAS THE EASY ELECTION
**117_1** | HE ALSO LAUNCHED A NEW STRATEGY FOR THE AGENCY | HE ALSO LAUNCHED A NEW STRATEGY FOR THE AGENCY
**217_1** | IT'S VERY DIFFICULT TO FIND A BUYER | IT'S VERY DIFFICULT TO FIND A BUYER
**317_1** | INSTEAD SHE JUST WIN AWAY | INSTEAD SHE JUST WENT AWAY
**417_1** | YOU CAN EXPECT A HELLO D A FUSE | YOU CAN EXPECT A HELL OF A FUSS
