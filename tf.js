import * as tf from '@tensorflow/tfjs'

// 1. 과거의 데이터를 준비합니다. 
const 온도 = [20, 21, 22, 23];
const 판매량 = [40, 42, 44, 46];
const 원인 = tf.tensor(온도);
const 결과 = tf.tensor(판매량);
console.log(원인.print());
console.log(결과.print());


// 2. 모델의 모양을 만듭니다. 
const X = tf.input({ shape: [1] });
const Y = tf.layers.dense({ units: 1 }).apply(X);
const model = tf.model({ inputs: X, outputs: Y });
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
model.compile(compileParam);

// 3. 데이터로 모델을 학습시킵니다. 
// const fitParam = { epochs: 6000 }
const fitParam = {
  epochs: 10,
  callbacks: {
    onEpochEnd: function (epoch, logs) {
      // console.log('epoch', epoch, logs, 'RMSE=>', Math.sqrt(logs.loss));
    }
  }
} // loss 추가 예제

console.log('학습중...');
model.fit(원인, 결과, fitParam).then(function (result) {

  // 4. 모델을 이용합니다. 
  // 4.1 기존의 데이터를 이용
  const 예측한결과 = model.predict(원인);
  예측한결과.print();

}).then(() => { console.log('학습 완료') }).then(() => {

  // 4.2 새로운 데이터를 이용
  const 다음주온도 = [15, 16, 17, 18, 19];
  const 다음주원인 = tf.tensor(다음주온도);
  const 다음주결과 = model.predict(다음주원인);
  다음주결과.print();
}).then(() => {
  const weights = model.getWeights();
  const result = Promise.all([model.predict(tf.tensor([20])).array(), weights[0].array(), weights[1].array()]);
  return result;
}).then(result => {
  console.log('predict : ', result[0][0][0]);
  console.log('weight : ', result[1][0][0]);
  console.log('bias : ', result[2][0]);
  console.log('calc : ', result[1][0][0] * 20 + result[2][0]);
}).then(() => {
  // model.save('downloads://my-model');
  model.save('localstorage://my-model4');
})

tf.loadLayersModel('localstorage://my-model').then(model => {
  model.predict(tf.tensor([20])).print();
});

// (async function () {
//   const fitParam = { epochs: 6000 };
//   console.log('학습중...');
//   const result = await model.fit(원인, 결과, fitParam);
//   const 예측한결과 = model.predict(원인);
//   예측한결과.print();
//   console.log('학습 완료');
// }())
