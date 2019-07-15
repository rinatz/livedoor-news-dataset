<script lang="ts">
import { Chart } from 'chart.js';
import { HorizontalBar, mixins } from 'vue-chartjs';
import { Component, Mixins, Prop, Watch } from 'vue-property-decorator';
import { Category } from '../Classification';

@Component
export default class CategoryChart extends Mixins(HorizontalBar, mixins.reactiveData) {
  @Prop({ default: [] }) private categories!: Category[];

  private chartData: Chart.ChartData = { datasets: [] };

  private readonly options: Chart.ChartOptions = {
    scales: {
      xAxes: [{ ticks: { min: 0, max: 100 } }],
    },
  };

  get backgroundColor() {
    let length = this.categories.length - 3;

    if (length < 0) {
      length = 0;
    }

    return [
      'rgba(0, 150, 136, 0.2)',
      'rgba(63, 81, 181, 0.2)',
      'rgba(233, 30, 99, 0.2)',
    ].concat(Array(length).fill('rgba(158, 158, 158, 0.2)'));
  }

  get borderColor() {
    let length = this.categories.length - 3;

    if (length < 0) {
      length = 0;
    }

    return [
      'rgba(0, 150, 136, 1)',
      'rgba(63, 81, 181, 1)',
      'rgba(233, 30, 99, 1)',
    ].concat(Array(length).fill('rgba(158, 158, 158, 1)'));
  }

  @Watch('categories')
  public onCategoriesChanged(newCategories: Category[], oldCategories: Category[]) {
    this.chartData = {
      labels: newCategories.map((x) => x.name),
      datasets: [
        {
          label: '信頼性 [%]',
          data: newCategories.map((x) => x.confidence * 100),
          borderWidth: 1,
          backgroundColor: this.backgroundColor,
          borderColor: this.borderColor,
        },
      ],
    };
  }
}
</script>
