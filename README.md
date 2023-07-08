# Protein-Secondary-structure-CNN-LSTM-classifier

Protein secondary structure prediction is a critical step in bioinformatics that seeks to identify the three-dimensional shape of a protein from its amino acid sequence. Accurately predicting these structures can offer valuable insights into the protein's function and the molecular mechanisms that underlie these functions. 

## Dataset 

The dataset used in this project comprises protein sequences and their corresponding secondary structure. Each protein sequence is represented as a string of characters, where each character represents a specific amino acid. Similarly, each protein's secondary structure is represented as a string of characters, where 'H' represents alpha helices, 'E' represents beta sheets, and 'C' represents random coils. Records with “unknown” proteins were dropped. 

 

## Sliding Window Calculation 

The sliding window calculation helps to simplify the problem of predicting a sequence by turning it into a series of fixed-size subsequence prediction problems. In the context of protein secondary structure prediction, a window size is first defined (for example, 15 amino acids). This means that at any given time, the model will consider only a subset of 15 consecutive amino acids from the total protein sequence. The window "slides" along the sequence, one amino acid at a time, and at each step, the model makes a prediction about the secondary structure of the central amino acid in the window. 

The sliding window method allows the model to consider the local context of each amino acid, i.e., the neighboring amino acids on either side. This is important because the secondary structure at a particular position in a protein is often influenced by the local sequence context. It also ensures that the input to the model is always of a fixed size, regardless of the length of the original protein sequence. 

 

## Encoding 

Orthogonal encoding is a method used to convert categorical variables, such as amino acids in a protein sequence, into a format that can be used by machine learning algorithms. It is a type of binary encoding, where each category is represented by a unique binary vector. 

  

 

In this project, each amino acid is represented by a 21-dimensional binary vector, where each dimension corresponds to one of the standard amino acids. For each amino acid, all elements of the vector are set to 0, except for the position corresponding to that amino acid, which is set to 1. 

For example, if 'A' is the first amino acid in our list, it could be represented as (1, 0, 0, ..., 0), 'B' as (0, 1, 0, ..., 0), and so on. 

This orthogonal encoding method allows the model to process the protein sequence as numerical data, while preserving the categorical nature of the amino acids. It also ensures that the model does not make any assumptions about the relationships between different amino acids based on their encoded values, as each amino acid is equally distant from all others in the encoding space. 

For sliding windows on edges of sequence added padding which consists of 2d matrixes of zeroes to better predict proteins’ beginnings and endings. 

 

## Measures SOV and Q3 

The accuracy of the prediction models was measured using Segment Overlap Measure (SOV) and Q3 scores. The Q3 score is a simple accuracy measure that compares the predicted and actual structures. It calculates the proportion of correctly predicted structures (helices, sheets, or coils) out of the total number of structures. 

On the other hand, the SOV is a more nuanced measure that takes into account not only the number of correct predictions but also the overlap of the correctly predicted segments. This is important in protein structure prediction, as even a small shift in the location of a structural element can significantly impact the protein's function. 

## Models' description. 

The problem of predicting secondary structure was addressed using model merged of two deep learning models: Convolutional Neural Network (CNN) and Long Short-Term Memory network (LSTM). 

  

 

 

## CNN Description 

Convolutional Neural Networks are designed to automatically and adaptively learn spatial hierarchies of features from the provided data. CNNs were originally developed for image analysis, but their capacity for feature learning and hierarchical representation makes them suitable for any multidimensional data, including orthogonal encoded protein sequences. In this work, the CNN was designed to take a window of amino acids at a time and output a prediction of secondary structure for the central amino acid of that window. 

 

## LSTM Description 

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) that are designed to remember past information and are therefore ideal for sequence prediction problems. In this study, an LSTM network was used to analyze the sequence of amino acids and predict their corresponding secondary structure. Similar to the CNN, the LSTM also uses a sliding window approach to focus on a section of the sequence at a time. 

 

## Model architecture 

Each deep learning model was trained independently. Data used to train model is quite small, so the models are simple to prevent overfitting. Architecture of each model: 

## CNN: 

 

There is one 2 dimensional convolutional layer with 16 filters, kernel size 3x3 and 1 channel. After convolutional layer there is flattening layer, converting output to the format proper for the final dense layer with softmax activation function classifying output into 3 classes. 

After a few experiments “adam” optimizer with default learning rate, 8 epochs, batch size – 64 (to decrease training time), validation split – 0.2, categorical crossentropy loss function are set. 

## LSTM: 

 

There is one LSTM layer with 64 LSTM units and one dense layer with softmax activation function classifying output into 3 classes.  

After few experiments “adam” optimizer with default learning rate, 8 epochs, batch size – 128 (to decrease training time), validation split – 0.2, categorical crossentropy loss function are set. 

## Results of experiments 

The prediction performance was then evaluated using external PERL script (http://dna.cs.miami.edu/SOV/), by comparing the predicted secondary structures with the actual structures in the validation set using Q3 and SOV scores. 

Coefficient values are modeled on the coefficients from the article: Protein secondary structure prediction based on integration of CNN and LSTM model. Additionally, two more experiments were conducted – first one with the swapped coefficient and second one with equal coefficients. For each coefficient pair, results for different sliding windows are provided.  

![obraz](https://github.com/karolkadlubowski/Protein-Secondary-structure-CNN-LSTM-classifier/assets/56251060/3ec77794-e4cc-4324-9500-7d1167db8c90)


The best results were received from parameters a = 0.42, b = 0.58 and with sliding window length of 21. It is expected result according to the mentioned article. Comparing the best obtained SOV and Q3 results to the other models, they seem quite good, especially taking into consideration limited computational power.  

For sliding window length of 15 the results are slightly worse, for sliding window length of 9 the results are siginificantly worse. There is a relationship that the larger the window, the higher the accuracy. In future research, a maximum properly working window size can be sought. 

In the future, the performance of these models could potentially be improved by optimizing and extending the network architecture, fine-tuning the hyperparameters, or using more advanced training techniques (eg. Cross-validation). Unfortunately, despite the authors’ awareness of possible upgrades, they could not be implemented due to limited computational power. 
