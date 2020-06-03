# Weakly_supervised_gender_ID
This directory contains the code for end-to-end neural network for weakly supervised gender ID from noisy speech 

CallFriend corpus (https://catalog.ldc.upenn.edu/LDC96S46) available from LDC is used for the experiments.

Steps:
1. Create an annotated dataset for gender ID from noisy speech using CallFriend corpus.
  a. A subset of callfriend conversations can be taken with various languages (12 available)
  b. Preprocess the data into single channel, multi-lingual, gender-balanced, short clips.
  c. Mix the conversational clips with real noisy database. We used DEMAND corpus (https://zenodo.org/record/1227121#.Xtf5LHUzY5k) to mix them. This assures that the conversations are simulated at various environmental conditions.
  d. Mixing is done at 0 dB, -5 dB and -10 dB, such as to illustrate the power of the denoiser. Hence, for severe noisy conditions, the aim is to obtain the gender information in a weakly supervised manner.
  
2. Train the DRNN denoiser (code is available at https://github.com/posenhuang/deeplearningsourceseparation) and generate the separated conversations.
3. Feed this to the training and testing scripts of raw-waveform based CNN network for gender classification.
4. Note that the splitting of the data needs to be done as specified in the IS18 paper: https://publications.idiap.ch/downloads/papers/2018/Sebastian_IS2018_2018.pdf

Sebastian J, Kumar M, DS PK, Magimai-Doss M, Murthy HA, Narayanan S. Denoising and Raw-waveform Networks for Weakly-Supervised Gender Identification on Noisy Speech. InInterspeech 2018 Jan 1 (pp. 292-296).
