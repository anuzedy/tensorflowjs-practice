import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { boston_cause, boston_result } from './boston';
import { boston_cause2, boston_result2 } from './boston2';


// 1. 과거의 데이터를 준비합니다. 
const 원인 = tf.tensor(boston_cause);
const 결과 = tf.tensor(boston_result);

// 2. 모델의 모양을 만듭니다. 
const X = tf.input({ shape: [13] });
const H1 = tf.layers.dense({ units: 13, activation: 'relu' }).apply(X); //hidden layer
const H2 = tf.layers.dense({ units: 13, activation: 'relu' }).apply(H1); //hidden layer
const Y = tf.layers.dense({ units: 1 }).apply(H2);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
model.compile(compileParam);

const surface = { name: '요약', tab: '모델' };
tfvis.show.modelSummary(surface, model);

// 3. 데이터로 모델을 학습시킵니다. 
//         const fitParam = {epochs: 100}
const _history = [];
const fitParam = {
  epochs: 200,
  callbacks: {
    onEpochEnd:
      function (epoch, logs) {
        console.log('epoch', epoch, logs, 'RMSE=>', Math.sqrt(logs.loss));
        const surface = { name: 'loss', tab: '역사' };
        _history.push(logs);
        tfvis.show.history(surface, _history, ['loss']);
      }
  }
} // loss 추가 예제
model.fit(원인, 결과, fitParam).then(function (result) {

  // 4. 모델을 이용합니다. 
  // 4.1 기존의 데이터를 이용
  const 예측한결과 = model.predict(원인);
  예측한결과.print();

}).then(() => {
  const weights = model.getWeights();
  return weights[0].array();
}).then(console.log);



// // 1. 과거의 데이터를 준비합니다.
// const 원인 = tf.tensor(boston_cause2);
// const 결과 = tf.tensor(boston_result2);

// // 2. 모델의 모양을 만듭니다.
// const X = tf.input({ shape: [12] });
// const Y = tf.layers.dense({ units: 2 }).apply(X);
// const model = tf.model({ inputs: X, outputs: Y });
// const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
// model.compile(compileParam);

// // 3. 데이터로 모델을 학습시킵니다.
// //         const fitParam = {epochs: 100}
// const fitParam = {
//   epochs: 200,
//   callbacks: {
//     onEpochEnd:
//       function (epoch, logs) {
//         console.log('epoch', epoch, logs, 'RMSE=>', Math.sqrt(logs.loss));
//       }
//   }
// } // loss 추가 예제
// model.fit(원인, 결과, fitParam).then(function (result) {

//   // 4. 모델을 이용합니다.
//   // 4.1 기존의 데이터를 이용
//   const 예측한결과 = model.predict(원인);
//   예측한결과.print();

// }).then(() => {
//   const weights = model.getWeights();
//   return Promise.all([weights[0].array(), weights[1].array()]);
// }).then(result => {
//   console.log(result[0]); //weight
//   console.log(result[1]); //bias
// });