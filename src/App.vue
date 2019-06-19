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
      <prediction-chart v-bind:predictions="predictions" v-bind:height="100"></prediction-chart>
      <tfidf-table v-bind:tfidf="tfidf"></tfidf-table>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import TfidfTable from "./components/TfidfTable.vue";
import PredictionChart from "./components/PredictionChart.vue";
import "bulma/css/bulma.min.css";

export default {
  name: "app",

  components: {
    TfidfTable,
    PredictionChart
  },

  data: function() {
    return {
      text: "",
      predictions: [],
      tfidf: []
    };
  },

  computed: {
    showResult: function() {
      return this.text;
    }
  },

  watch: {
    text: function(text) {
      axios
        .post("/classification", { text })
        .then(({ data: { predictions, tfidf } }) => {
          this.$nextTick(function() {
            this.predictions = predictions;
            this.tfidf = tfidf;
          });
        });
    }
  }
};
</script>
