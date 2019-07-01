<template>
  <table class="table is-striped is-fullwidth">
    <thead>
      <tr>
        <th>ランク</th>
        <th>ワード</th>
        <th>TF-IDF</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="(token, index) in tokens" v-bind:key="index">
        <td>{{ index + 1 }}</td>
        <td>{{ token.lemma }}</td>
        <td>{{ token.tfidf }}</td>
      </tr>
    </tbody>
  </table>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Token } from '../typings';
import 'bulma/css/bulma.min.css';

@Component
export default class TokenTable extends Vue {
  @Prop({ default: [] }) private tokens!: Token[];
}
</script>
