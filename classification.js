import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import * as dfd from 'danfojs';

dfd.readCSV('https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv').then(function (data) {
  console.log(data);
  data.print();
  const 독립변수 = data.loc({ columns: ['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭'] });
  독립변수.print();
  const encoder = new dfd.OneHotEncoder();
  const 종속변수 = encoder.fit(data['품종']).transform(data['품종']);
  data['품종'].print();
  종속변수.print();

  const X = tf.input({ shape: [4] });
  const H = tf.layers.dense({ units: 4, activation: 'relu' }).apply(X);
  // const Y = tf.layers.dense({ units: 3 }).apply(H);
  const Y = tf.layers.dense({ units: 3, activation: 'softmax' }).apply(H);
  const model = tf.model({ inputs: X, outputs: Y });

  // const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
  const compileParam = { optimizer: tf.train.adam(), loss: 'categoricalCrossentropy', metrics: ['accuracy'] }

  model.compile(compileParam);

  tfvis.show.modelSummary({ name: '요약', tab: '모델' }, model);

  const _history = [];
  const fitParam = {
    epochs: 200,
    callbacks: {
      onEpochEnd:
        function (epoch, logs) {
          console.log('epoch', epoch, logs, 'RMSE=>', Math.sqrt(logs.loss));
          _history.push(logs);
          tfvis.show.history({ name: 'loss', tab: '역사' }, _history, ['loss']);
          tfvis.show.history({ name: 'accuracy', tab: '역사' }, _history, ['acc']);
        }
    }
  }

  model.fit(독립변수.tensor, 종속변수.tensor, fitParam).then(function (result) {
    //         // 4. 모델을 이용합니다. 
    //         // 4.1 기존의 데이터를 이용
    const 예측한결과 = new dfd.DataFrame(model.predict(독립변수.tensor));
    예측한결과.print();
    종속변수.print();
  });
})