# Product-review-analysis
The most common form of sentiment analysis is sentence-level sentiment analysis. This is an interesting field of text analysis which can classify text to positive, negative or neutral category. A more fine-grained version of sentiment analysis is the aspect level sentiment analysis. In this, each text is parsed to extract features or aspects from it and detect polarity of those aspects. Much research has been carried out in the past decade and many methods are developed to extract aspect terms. Many papers only focus on the aspect term extraction task since it has not achieved impressive accuracy yet. In this paper, sentence level sentiment analysis is first implemented and then the knowledge of the same is carried on, to work with aspect level sentiment analysis.

Convert text to lowercase -> tokenize -> POS tagging -> dependecy parsing -> collect all nouns and adjectives -> make feature and opinion pairs using dependency relation

Run it as:

nlp = stanfordnlp.Pipeline()
stop_words = set(stopwords.words('english'))
txt = "The picture quality is awesome but the battery life is poor."

print(product_review_analysis(txt, stop_words, nlp))

Output will look like:
[['picturequality', ['awesome']], ['batterylife', ['poor']]]
