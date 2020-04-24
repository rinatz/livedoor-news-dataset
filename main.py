import streamlit as st
from bokeh.plotting import figure

from news_classifier import preprocessing, livedoor_news, training


def main():
    st.title("ニュースの分類器")

    mecab = preprocessing.MeCabTokenizer()
    tokenizer = livedoor_news.get_tokenizer()
    model = training.load_model()

    text = st.text_area("文章を入力してください。")

    if text:
        texts = [mecab.tokenize(text)]
        tfidf = tokenizer.texts_to_matrix(texts, mode="tfidf")
        confidences = model.predict(tfidf)

        classes = list(livedoor_news.get_classes().values())
        sorted_classes = sorted(classes, key=lambda x: confidences[0][classes.index(x)])

        p = figure(y_range=sorted_classes, title="信頼性 [%]")
        p.hbar(y=classes, right=confidences[0])
        st.bokeh_chart(p)


if __name__ == "__main__":
    main()
