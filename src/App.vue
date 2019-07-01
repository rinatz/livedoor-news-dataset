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

<script lang="ts">
import axios from 'axios';
import { Component, Prop, Watch, Vue } from 'vue-property-decorator';
import CategoryChart from './components/CategoryChart.vue';
import TokenTable from './components/TokenTable.vue';
import { Category, Token, Classification } from './typings';
import 'bulma/css/bulma.min.css';

@Component({
  components: {
    CategoryChart,
    TokenTable,
  },
})
export default class App extends Vue {
  private text: string = '';
  private categories: Category[] = [];
  private tokens: Token[] = [];

  get showResult(): boolean {
    return !!this.text;
  }

  @Watch('text')
  public onTextInput(newText: string, oldText: string): void {
    if (!newText) {
      return;
    }

    axios
      .post<Classification>('/classifications', { text: newText })
      .then(({ data: { categories, tokens } }) => {
        this.$nextTick(() => {
          this.categories = categories;
          this.tokens = tokens;
        });
      });
  }
}
</script>
