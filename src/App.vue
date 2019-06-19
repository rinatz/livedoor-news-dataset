<template>
  <div class="container">
    <div class="field">
      <label class="label">入力テキスト</label>
      <div class="control">
        <textarea class="textarea" placeholder="ここにテキストを入力してください" rows="5" v-model="text"></textarea>
      </div>
    </div>

    <div v-if="showResult">
      <label class="label">判定結果</label>

      <horizontal-bar-chart
        v-bind:chart-data="chartData"
        v-bind:options="chartOptions"
        v-bind:height="100"
      ></horizontal-bar-chart>

      <table class="table is-striped is-fullwidth">
        <thead>
          <tr>
            <th>ランク</th>
            <th>ワード</th>
            <th>TF-IDF</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(x, index) in tfidf" v-bind:key="index">
            <td>{{ index + 1 }}</td>
            <td>{{ x.word }}</td>
            <td>{{ x.value }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import HorizontalBarChart from "./components/HorizontalBarChart.vue";
import "bulma/css/bulma.min.css";

export default {
  name: "app",
  components: {
    HorizontalBarChart
  },
  data: function() {
    return {
      text: "",
      showResult: false,
      chartData: {},
      chartOptions: {
        scales: {
          xAxes: [{ ticks: { min: 0, max: 100 } }]
        }
      },
      tfidf: []
    };
  },
  watch: {
    text: function(text) {
      if (!text) {
        this.showResult = false;
        return;
      }

      axios
        .post(`http://${location.host}/scores`, { text })
        .then(({ data: { classification, tfidf } }) => {
          let length = classification.length - 3;

          if (length < 0) {
            length = 0;
          }

          const backgroundColor = [
            "rgba(0, 150, 136, 0.2)",
            "rgba(63, 81, 181, 0.2)",
            "rgba(233, 30, 99, 0.2)"
          ].concat(Array(length).fill("rgba(158, 158, 158, 0.2)"));

          const borderColor = [
            "rgba(0, 150, 136, 1)",
            "rgba(63, 81, 181, 1)",
            "rgba(233, 30, 99, 1)"
          ].concat(Array(length).fill("rgba(158, 158, 158, 1)"));

          this.showResult = true;

          this.chartData = {
            labels: classification.map(x => x["description"]),
            datasets: [
              {
                backgroundColor,
                borderColor,
                label: "確率 [%]",
                data: classification.map(x => x["score"] * 100),
                borderWidth: 1
              }
            ]
          };

          this.tfidf = tfidf;
        });
    }
  }
};
</script>
