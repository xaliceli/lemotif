# Lemotif: Affective Visual Journal

[![DOI](https://zenodo.org/badge/183519985.svg)](https://zenodo.org/badge/latestdoi/183519985)

[Demo](http://lemotif.cloudcv.org) | [Paper](https://arxiv.org/abs/1903.07766)

We present Lemotif, an integrated natural language processing and image generation system that uses machine learning to (1) parse a text-based input journal entry describing the user’s day for salient themes and emotions and (2) visualize the detected themes and emotions in creative and appealing image motifs. Synthesizing approaches from artificial intelligence and psychology, Lemotif acts as an affective visual journal, encouraging users to regularly write and reflect on their daily experiences through visual reinforcement. 

By making patterns in emotions and their sources more apparent, Lemotif aims to help users better understand their emotional lives, identify opportunities for action, and track the effectiveness of behavioral changes over time. We verify via human studies that prospective users prefer motifs generated by Lemotif over corresponding baselines, find the motifs representative of their journal entries, and think they would be more likely to journal regularly using a Lemotif-based app.

## Sample Outputs

![Sample Outputs](https://github.com/xaliceli/lemotif/blob/master/assets/docs/sample.png)

## Data & Models

### Dataset

Our [dataset](https://github.com/xaliceli/lemotif/blob/master/assets/data/lemotif-data-cleaned-flat.csv) contains 1,473 text samples with topic and emotion labels from the same Amazon MTurk respondents who wrote each journal entry. This dataset was manually cleaned to omit nonsensical and/or nonresponsive entries. Each text sample contains one positive topic label and up to multiple positive emotion labels.

### Trained Models

* Autoencoder: Keras .h5 files can be found [here](https://github.com/xaliceli/lemotif/blob/master/app/models/ae/).
* NLP parser: Tensorflow checkpoint can be found [here](https://drive.google.com/open?id=1-2PP0fk5_33qu0Lhyt3azFhAJTkIszmA).

## Future Work

* There is a lot of room to improve upon the existing NLP model with additional data and integrating common-sense knowledge around topic mappings, synonyms, known word-sentiment associations, and more. 
* The existing autoencoder can be improved and expanded through better architectures, parameter tuning, and training on different data representing different artistic movements. 
* We didn't achieve satisfactory results using a conditional GAN for the duration of this project, but better architectures, data, and more parameter tuning would likely yield fruitful results on this front.

## Citation


### Paper
```
@misc{li2019lemotif,
    title={Lemotif: An Affective Visual Journal Using Deep Neural Networks},
    author={X. Alice Li and Devi Parikh},
    year={2019},
    eprint={1903.07766},
    archivePrefix={arXiv},
    primaryClass={cs.HC}
}
```

### Code
```
@misc{x_alice_li_2019_3269893,
  author       = {X. Alice Li and Devi Parikh},
  title        = {Lemotif: Affective Visual Journal},
  month        = jul,
  year         = 2019,
  doi          = {10.5281/zenodo.3269893},
  url          = {https://doi.org/10.5281/zenodo.3269893}
}
```
