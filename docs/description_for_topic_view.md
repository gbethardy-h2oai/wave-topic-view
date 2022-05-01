# Topic View
Visually explore topic and word distributions from LDA topic analysis.

## Current Functionality

### Select the Dataset to upload
* Choose the csv file to upload (required).
* Upload a csv file, 50MB maximum size.

### Select LDA topic model parameters
* Choose the text column model (required).
* Select text preprocessing, including stop word removal, stemming, and lemmatization.
* Optionally enter additional stop words to augment the standard English stop words.
* Select the number of topics.

### Generate the topic model and metrics
* Review the table of topic/word distributions .
* Review the coherence and perplexity measures.

### Visually explore the topic/word distributions
* Visually explore the topic clusters in a 2-d space using multidimensional scaling.
* Select a topic, change the lambda relevance value, and view its effect on the importance of the words within the topic.
* Change the model parameters, e.g. the number of topics, etc. and investigate how the model quality changes.

*Have feedback? Reach out to us on slack.

*The Topic View app is example code from H2O for how to visually explore LDA topic model results.
*The goal is to help accelerate with building a front end for NLP analysis.
*This is not a fully supported product. We would love your feedback on the ease of making
*this app your own, and the content of the app. We are open to requests on this app but
*please note that since it is demo code the regular H2O Support process does not apply.