# RecallNet
We propose a novel model called RecallNet for speech enhancement.
Prosposed RecallNet allows to listen to the noisy speech for multiple times and store the information each listening in the external memory.
The stored information will be attended over during the last time of listening and help to enhance the quality of speech.
In our experients, we consider only two times of listening. 

<img src="Others/RecallNet.png" width="50%">

## Setting
- Hardware:
	- CPU: Intel Core i7-4930K @3.40 GHz
	- RAM: 64 GB DDR3-1600
	- GPU: NVIDIA Tesla K20c 6 GB RAM
- Tensorflow 0.12
- Dataset
	- Wall Street Journal Corpus
	- Noises are collected from [freeSFX](http://www.freesfx.co.uk/soundeffects/) and [AudioMicro](http://www.audiomicro.com/free-sound-effects)

## Result
- An example of mixed signal and demixed signals by DNN, LSTM, NTM and RecallNet

|<img src="Others/mix.png" width="50%">|<img src="Others/clean.png" width="50%">|
|:------------------------------------:|:--------------------------------------:|
|Mix signal                            |Clean signal                            |
|<img src="Others/recall.png" width="50%">|<img src="Others/NTM.png" width="50%">|
|:---------------------------------------:|:------------------------------------:|
|Demixed signal (RecallNet)               |Demixed signal (NTM)                  |
|<img src="Others/LSTM.png" width="50%">|<img src="Others/DNN.png" width="50%">|
|:------------------------:|:--------------------------:|
|Demixed signal (LSTM)     |Demixed signal (DNN)        |

- STOI measure on bus and caf noises

|<img src="Others/stoi1.png">|<img src="Others/stoi2.png">|
|:------------------------:|:--------------------------:|
|Seen speakers             |Unseen speakers             |

