from bokeh.plotting import figure
import streamlit as st

import livedoor


def main():
    model = livedoor.load_model()

    st.title("ニュースの分類器")
    text = st.text_area("文章を入力してください。")

    if text:
        categories = model.predict(text)

        chart = figure(y_range=categories.site_name, title="信頼性 [%]")
        chart.hbar(y=categories.site_name, right=categories.confidence)

        st.bokeh_chart(chart)


if __name__ == "__main__":
    main()
