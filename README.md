## Dream Cluster
### An approach to dream analysis using LDA topic modeling and NLP
**Connor Eaton - February 2019**

### Overview

Humans dream. Sometimes our dreams are wild subconscious utterances drenched in symbolism. Sometimes they are as mundane as eating a sandwhich. Regardless, as mysterious as the source of our dreams are, even more puzzling is how across cultures, humans often dream similar dreams. 

As our modern culture has made large efforts to sever itself from our symbolic roots, dreams are not treated with the same respect as the more objective facets of our lives. From a lack of time and interest or from embarrasment and shame, many of us ignore our dreams, casting aside a valuable tool for self-understanding. However, if we use modern techniques to conveniently dive into the ancient depths of our dreams, we may be able to bring actionable insights back to our day-to-day life. This was the inspiration for my project.

My project has two parts. This first aim was to scrape dreams from the most active Reddit subforums for the self-reporting of dreams. From this text data, I built a corpus of dreams and used unsupervised learning to find interpretatble topics to effectively build 11 dream categories. The second aim was to build a pipeline to pass in a new dream, preprocess the text, predict its topic, and return the three most similar dreams from within the original corpus. Such insight will provide the user with a new framework for understanding their dreams and perhaps build a community of like-minded dreamers seeking to better understand themselves and eachother.

### Data

Using the PRAW package, data was scraped from a multitude of Reddit subforums where users self-report dreams. The scripts used for this process can be found in the src/data_collection/ directory. External, combined raw, and processed data can be found in the data/ directory and the preprocessing scripts can be found in src/features/ directory. In total, there were 4,572 dreams collected and processed.

### Tools

Many tools were used for this project, including python, pandas, numpy, sklearn, gensim, PRAW, spacy, and NLTK. For a full list of depencies, see /requirements.txt.

### Model and Results

Latent Dirichlet Allocation (LDA) was chosen for its ability to create human interpretable topics from a corpus of text. To optimize for subjective topic interpretation and coherence score, the topic number was set to 11. Some of the topics generated were: **flying/water dreams, sickness/health dreams, exploring mysterious place dreams, fighting dreams, and school dreams**. 

[![Screen-Shot-2019-03-02-at-11-11-59-AM.png](https://i.postimg.cc/6Qq6MNgJ/Screen-Shot-2019-03-02-at-11-11-59-AM.png)](https://postimg.cc/XZM0vtnQ)

After training, new dreams were passed into the model to predict their topic probabilty. In addition, sklearn's TfIdf Vectorizer was used to find the 3 most cosine similar dreams within the corpus to the new dream. See src/models/ for the jupyter notebooks for this process. 

The final product (for now), is a python script that when run, accepts a dream input by the user in the form of raw text, completes all of the necassary text preprocessing, predicts the topic probabilty of the dream, and returns the 3 most similar dreams from within the corpus. As all of these processes are quickly computed by one script (see src/app/main.py), I have laid the foundations for a flask app to built. 

### Future Work

There is much room for growth in this project and it is nowhere near completion. One of the major areas for growth is correcting topic imbalance. As you can see from the tSNE plot below, there is clear seperation between many of the topics, but some are much bigger than others.

[![Screen-Shot-2019-03-01-at-8-45-26-AM.png](https://i.postimg.cc/yNtQ2FQ6/Screen-Shot-2019-03-01-at-8-45-26-AM.png)](https://postimg.cc/06G0wJ1h)

The first step in correcting this imblance will be collecting more data. As time passes, I will continue to update my corpus with new dreams from recent Reddit submissions, as well as incorporate user input dreams into the main corpus. This, along with improving the inclusion and/or removal of certain words from my corpus, will likely correct topic imbalance and improve clarity of interpretation. I also intend to build a flask app from this project so that anyone can gain insight on their dreams and ultimately, better understand themselves at a deep level.
