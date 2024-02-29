# DeepForcedAligner for Text-to-Speech

<!-- [![codecov](https://codecov.io/gh/roedoejet/g2p/branch/master/graph/badge.svg)](https://codecov.io/gh/roedoejet/g2p) -->
<!-- [![Build Status](https://github.com/roedoejet/g2p/actions/workflows/tests.yml/badge.svg)](https://github.com/roedoejet/g2p/actions) -->
<!-- [![PyPI package](https://img.shields.io/pypi/v/hifigan.svg)](https://pypi.org/project/g2p/) -->
[![license](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/roedoejet/DeepForcedAligner)

🚧 Under Construction! This repo is not expected to work fully. Please check back later for a stable release. 🚧

> A fork of the DeepForcedAligner project implemented in PyTorch Lightning

From the original authors:

> With this tool you can create accurate text-audio alignments given a bunch of audio files and their transcription. The alignments can for example be used to train text-to-speech models such as
[FastSpeech](https://arxiv.org/abs/1905.09263?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529). In comparison to other forced alignment tools this repo has following advantages:

>  - **Multilingual:** By design, the DFA is language-agnostic and can align both characters or phonemes.
>  - **Robustness:** The alignment extraction is highly tolerant against text errors and silent characters.
>  - **Convenience:** Easy installation with no extra dependencies. You can provide your own data in the standard LJSpeech format without special preprocessing (such as applying phonetic dictionaries, non-speech annotations etc.).

> The approach is based on training a simple speech recognition model with CTC loss on mel spectrograms extracted from the wav files.


This repo has been separated in case you would like to use it separately from the broader SGILE system, but if you are looking to build speech synthesis systems from scratch, please visit [the main repository](https://github.com/roedoejet/EveryVoice)

## Table of Contents
- DeepForcedAligner
  - [Table of Contents](#table-of-contents)
  - [Background](#background)
  - [Install](#install)
  - [Usage](#usage)
  <!-- - [How to Cite](#citation)
  - [License](#license) -->

See also:
  - [SGILE FastSpeech2](https://github.com/roedoejet/FastSpeech2_lightning)
  - [SGILE Vocoder](https://github.com/roedoejet/HiFiGAN_iSTFT_lightning)
  - [Requirements and Motivations of Low-Resource Speech Synthesis for Language Revitalization](https://aclanthology.org/2022.acl-long.507/)

## Background

There are approximately 70 Indigenous languages spoken in Canada from 10 distinct language families.  As a consequence of the residential school system and other policies of cultural suppression, the majority of these languages now have fewer than 500 fluent speakers remaining, most of them elderly.

Despite this, Indigenous people have resisted colonial policies and continued speaking their languages, with interest by students and parents in Indigenous language education continuing to grow. Teachers are often overwhelmed by the number of students, and the trend towards online education means many students who have not previously had access to language classes now do. Supporting these growing cohorts of students comes with unique challenges in languages with few fluent first-language speakers. Teachers are particularly concerned with providing their students with opportunities to hear the language outside of class.

While there is no replacement for a speaker of an Indigenous language, there are possible applications for speech synthesis (text-to-speech) to supplement existing text-based tools like verb conjugators, dictionaries and phrasebooks.

The National Research Council has partnered with the Onkwawenna Kentyohkwa Kanyen’kéha immersion school, W̱SÁNEĆ School Board, University nuhelot’įne thaiyots’į nistameyimâkanak Blue Quills, and the University of Edinburgh to research and develop state-of-the-art speech synthesis (text-to-speech) systems and techniques for Indigenous languages in Canada, with a focus on how to integrate text-to-speech technology into the classroom.

## Installation

Clone clone the repo and pip install it locally:

```sh
$ git clone https://github.com/roedoejet/DeepForcedAligner.git
$ cd DeepForcedAligner
$ pip install -e .
```

## Usage

### Configuration

It is recommended to install `everyvoice` and run `everyvoice new-project` to create your configuration. After running the command, a `config` folder will be created with an `everyvoice-aligner.yaml` configuration file.

### Preprocessing

Preprocess by running: `dfaligner preprocess config/everyvoice-aligner.yaml` to generate the Mel spectrograms and audio and text representations required for the model using the base configuration.

To run only a subset of the steps, use `dfaligner preprocess config/everyvoice-aligner.yaml -s text` to only run text preprocessing for example

### Training

Train by running `dfaligner train config/everyvoice-aligner.yaml` to use the base configuration.

You can pass updates to the configuration through the command line like so:

`dfaligner train config/everyvoice-aligner.yaml -c preprocessing.save_dir=/my/new/path -c training.batch_size=16`

### Alignment Extraction

To extract alignments from the model run `dfaligner extract-alignments config/everyvoice-aligner.yaml --model path/to/model.ckpt`.

## Contributing

Feel free to dive in!
 - [Open an issue](https://github.com/roedoejet/EveryVoice/issues/new) in the main EveryVoice repo with the tag `[DeepForceAligner]`,
 - submit PRs to this repo with a corresponding submodule update PR to [EveryVoice](https://github.com/roedoejet/EveryVoice).

This repo follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

You can install our standard Git hooks by running these commands in your sandbox:

```sh
pip install -r requirements.dev.txt
pre-commit install
gitlint install-hook
```

Have a look at [Contributing.md](https://github.com/roedoejet/EveryVoice/blob/main/Contributing.md)
for the full details on the Conventional Commit messages we prefer, our code
formatting conventions, and our Git hooks.

You can then interactively install the package by running the following command from the project root:

```sh
pip install -e .
```


## Acknowledgements

This project is only possible because of the work of the authors of [DeepForcedAligner](https://github.com/as-ideas/DeepForcedAligner) (Christian Schäfer and Francisco Cardinale). Please cite and start their work.
