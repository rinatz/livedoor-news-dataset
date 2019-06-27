<script>
import { HorizontalBar, mixins } from 'vue-chartjs';

export default {
  extends: HorizontalBar,
  mixins: [mixins.reactiveData],

  props: {
    categories: {
      type: Array,
      required: false,
      default: () => []
    }
  },

  data: function() {
    return {
      options: {
        scales: {
          xAxes: [{ ticks: { min: 0, max: 100 } }]
        }
      }
    };
  },

  computed: {
    backgroundColor: function() {
      let length = this.categories.length - 3;

      if (length < 0) {
        length = 0;
      }

      return [
        'rgba(0, 150, 136, 0.2)',
        'rgba(63, 81, 181, 0.2)',
        'rgba(233, 30, 99, 0.2)'
      ].concat(Array(length).fill('rgba(158, 158, 158, 0.2)'));
    },
    borderColor: function() {
      let length = this.categories.length - 3;

      if (length < 0) {
        length = 0;
      }

      return [
        'rgba(0, 150, 136, 1)',
        'rgba(63, 81, 181, 1)',
        'rgba(233, 30, 99, 1)'
      ].concat(Array(length).fill('rgba(158, 158, 158, 1)'));
    }
  },

  watch: {
    categories: function(categories) {
      this.chartData = {
        labels: categories.map(x => x['name']),
        datasets: [
          {
            label: '信頼性 [%]',
            data: categories.map(x => x['confidence'] * 100),
            borderWidth: 1,
            backgroundColor: this.backgroundColor,
            borderColor: this.borderColor
          }
        ]
      };
    }
  }
};
</script>
