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
      <tr v-for="(x, index) in tfidf" v-bind:key="index">
        <td>{{ index + 1 }}</td>
        <td>{{ x.word }}</td>
        <td>{{ x.value }}</td>
      </tr>
    </tbody>
  </table>
</template>

<script>
import "bulma/css/bulma.min.css";

export default {
  props: ["tfidf"]
};
</script>
