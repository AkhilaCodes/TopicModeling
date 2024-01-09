# Install
#!pip install pyLDAvis -qq
#!pip install -qq -U gensim
#!pip install spacy -qq
#!pip install matplotlib -qq
#!pip install seaborn -qq
#!python -m spacy download en_core_web_md -qq

# Import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pyLDAvis.gensim_models
#pyLDAvis.enable_notebook()# Visualise inside a notebook
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # Read the data
    reports = pd.read_csv('/Users/akhilajoshi/Python_Coding/topic_m/cordis-h2020reports.gz')
    reports.head()
    reports.info()

    # Our spaCy model:
    nlp = en_core_web_md.load()

    # Tags I want to remove from the text
    removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
    tokens = []

    for summary in nlp.pipe(reports['summary']):
        proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
        tokens.append(proj_tok)

    # Add tokens to a new column
    reports['tokens'] = tokens
    reports['tokens']

    # Create dictionary
    # I will apply the Dictionary Object from Gensim, which maps each word to their unique ID:
    dictionary = Dictionary(reports['tokens'])
    print(dictionary.token2id)

    # Filter dictionary
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)

    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in reports['tokens']]

    # LDA model building
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=10, workers = 4, passes=10)

    # Coherence score using C_umass:
    topics = []
    score = []
    for i in range(1,20,1):
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers = 4, passes=10, random_state=100)
        cm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        topics.append(i)
        score.append(cm.get_coherence())
    _=plt.plot(topics, score)
    _=plt.xlabel('Number of Topics')
    _=plt.ylabel('Coherence Score')
    plt.title('Coherence score using C_umass')
    plt.show()
    plt.savefig('C_umass.png')

    # Coherence score using C_v:
    topics = []
    score = []
    for i in range(1,20,1):
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers = 4, passes=10, random_state=100)
        cm = CoherenceModel(model=lda_model, texts=reports['tokens'], corpus=corpus, dictionary=dictionary, coherence='c_v')
        topics.append(i)
        score.append(cm.get_coherence())
    _=plt.plot(topics, score)
    _=plt.xlabel('Number of Topics')
    _=plt.ylabel('Coherence Score')
    plt.title('Coherence score using C_v')
    plt.show()
    plt.savefig('C_v.png')

    # Optimal model
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=100, num_topics=5, workers = 4, passes=100)

    # Print topics
    lda_model.print_topics(-1)

    # Where does a text belong to
    lda_model[corpus][0]
    reports['summary'][0]

    # Visualize topics
    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.display(lda_display)

    # Save the report
    pyLDAvis.save_html(lda_display, 'index.html')

if __name__ == "__main__":
    main()
