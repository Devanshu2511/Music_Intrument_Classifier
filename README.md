# Music_Intrument_Classifier
This is a simple classifier that is able to detect single-note sounds of various musical instruments. Currently supported types are guitar, mandolin and trumpet. The audios are supposed to be single-note sounds.

# Dataset
The dataset used to train this classifier was collected from London Philharmonic Orchestra Dataset (http://www.philharmonia.co.uk/explore/sound_samples). Each audio file records one note from one of the five instruments, and has a length from 0.25 seconds to 6 seconds.

# Approach
Used the PCA approach created from scratch. Trained the dataset on our PCA model and obtained the recognization results with an accuracy of 70%.
