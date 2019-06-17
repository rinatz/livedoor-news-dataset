<template>
  <div class="container">
    <div class="field">
      <label class="label">入力テキスト</label>
      <div class="control">
        <textarea class="textarea" placeholder="ここにテキストを入力してください" rows="5" v-model="text"></textarea>
      </div>
    </div>

    <div>
      <label class="label">判定結果</label>
      <horizontal-bar-chart
        v-if="showChart"
        v-bind:chart-data="chartData"
        v-bind:options="chartOptions"
        v-bind:height="100"
      ></horizontal-bar-chart>
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
      showChart: false,
      chartData: {},
      chartOptions: {
        scales: {
          xAxes: [{ ticks: { min: 0, max: 100 } }]
        }
      }
    };
  },
  watch: {
    text: function() {
      if (!this.text) {
        this.showChart = false;
        return;
      }

      axios
        .post(`http://${location.host}/scores`, { text: this.text })
        .then(({ data: { scores } }) => {
          let length = scores.length - 3;

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

          this.showChart = true;

          this.chartData = {
            labels: scores.map(x => x["description"]),
            datasets: [
              {
                backgroundColor,
                borderColor,
                label: "確率 [%]",
                data: scores.map(x => x["value"] * 100),
                borderWidth: 1
              }
            ]
          };
        });
    }
  }
};
</script>
