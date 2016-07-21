"""Main module to work with the cnn."""
from nlp.cnn0 import cnn
from nlp.dataPreprocessing import buildInputData, buildInputData_twitter
import os


def classify(path_sortie_scraping, id_presta, lang='english', dtype='sentence'):
    """Classification with the neural net."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    ##########################################################################################
    n_classes = 0
    cnn_model = ''
    w2v_model = ''

    if lang == 'en':
        lang = 'english'
        if dtype == 'sentence':
            cnn_model = 'models/cnn-model-rt-80epoch.cpkl'
            w2v_model = 'models/300features-GoogleNews.w2v'
            n_classes = 2
            path = './data/classification/sentence/'
        elif dtype == 'twitter':
            cnn_model = 'models/cnn-model-twitter-80epoch.cpkl'
            w2v_model = 'models/300features-twitter-dataset.w2v'
            n_classes = 2
            path = './data/classification/twitter/'
        elif dtype == 'reviews':
            # TODO
            pass
    elif lang == 'fr':
        lang = 'french'
        if dtype == 'sentence':
            # TODO
            pass
        if dtype == 'twitter':
            # TODO
            pass
        if dtype == 'reviews':
            # TODO
            pass

    if dtype == 'twitter':
        x, y, sentences_list_str = buildInputData_twitter(path_sortie_scraping, w2v_model, n_classes=n_classes, lang=lang)
    else:
        x, y, sentences_list_str = buildInputData(path_sortie_scraping, w2v_model, n_classes=n_classes, lang=lang)
    seq_len = len(x[0])
    print("\nMax len sequence : %s" % seq_len)
    print("Word2Vec model : %s" % w2v_model)
    print("Cnn session : %s" % cnn_model)
    ##########################################################################################
    myCnn = cnn(seq_len=seq_len, n_classes=n_classes)
    myCnn.load(cnn_model)
    binary_predictions = myCnn.classify_binary(x, y)
    pos_count = 0
    neg_count = 0
    for pred in binary_predictions:
        if pred == 'neg':
            neg_count += 1
        elif pred == 'pos':
            pos_count += 1

    print("POS / NEG :")
    print(pos_count, neg_count)
    max_index = myCnn.getMaxSentiments(x, y)
    max_sent_sentences = [sentences_list_str[mx_i[1]] for mx_i in max_index]
    print("Max sentences : ")
    print(max_sent_sentences)

    print("\nWriting the classification results in .txt files ...")
    f = open(path + 'out_classification_count_' + str(id_presta) + ".txt", 'w', encoding='utf-8')
    f.write("pos,%s\n" % pos_count)
    f.write("neg,%s\n" % neg_count)
    f.close()
    f = open(path + 'out_classification_sent_' + str(id_presta) + ".txt", 'w', encoding='utf-8')
    f.write("max_pos,%s" % max_sent_sentences[1])
    f.write("max_neg,%s" % max_sent_sentences[0])
    f.close()
    myCnn.reset()
