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
      <category-chart v-bind:categories="categories" v-bind:height="100"></category-chart>
      <token-table v-bind:tokens="tokens"></token-table>
    </div>
  </div>
</template>

<script>
import axios from "axios";
import CategoryChart from "./components/CategoryChart.vue";
import TokenTable from "./components/TokenTable.vue";
import "bulma/css/bulma.min.css";

export default {
  name: "app",

  components: {
    CategoryChart,
    TokenTable
  },

  data: function() {
    return {
      text: "",
      categories: [],
      tokens: []
    };
  },

  computed: {
    showResult: function() {
      return this.text;
    }
  },

  watch: {
    text: function(text) {
      if (!this.text) {
        return;
      }

      axios
        .post("/classifications", { text })
        .then(({ data: { categories, tokens } }) => {
          this.$nextTick(function() {
            this.categories = categories;
            this.tokens = tokens;
          });
        });
    }
  }
};
</script>
