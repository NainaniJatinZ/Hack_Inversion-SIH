let myChart = document.getElementById('myChart').getContext('2d');

// Global Options
Chart.defaults.global.defaultFontFamily = 'Lato';
Chart.defaults.global.defaultFontsize = 18;
Chart.defaults.global.defaultFontColor = '#777';


let PrPopChart = new Chart(myChart, {
  type:'line', // bar, horizontalBar, pie, line, doughnut, radar, polarArea
  data:{
    labels: dates,
    datasets:[{
      label:'Close',
      data: closex,

      // backgroundColor: 'green',
      backgroundColor:[
        '#e65100',
        '#e65100',
        '#f57c00',
        '#fb8c00',
        '#ff9800',
        '#ffa726',
        '#ffb74d',
        '#ffcc80',
        '#ffe0b2',
        '#fff3e0'
      ],
      borderWidth:1,
      borderColor:'white',
      hoverBorderWidth: 3,
      hoverBorderColor: 'grey'
    }]
  },
  options:{
    title:{
      display: true,
      text: 'Largest Municipalities in Puerto Rico',
      fontSize:25
    },
    legend:{
      display:true,
      position:'right',
      labels:{
        fontColor: '#000'
      }
    },
    layout:{
      padding:{
        left:50,
        right:0,
        bottom:0,
        top:0
      }
    },
    tooltips:{
      enabled:true
    }
  }
});