const tf = require('@tensorflow/tfjs');

// 간단한 선형 회귀 모델: y = 2x + 1을 학습
async function trainModel() {
  // 모델 정의
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // 손실 함수와 옵티마이저 설정
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // 학습 데이터
  const xs = tf.tensor1d([1, 2, 3, 4]);
  const ys = tf.tensor1d([3, 5, 7, 9]); // 실제 y = 2x + 1

  // 모델 학습
  await model.fit(xs, ys, {
    epochs: 200,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 50 === 0) {
          console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
        }
      }
    }
  });

  // 테스트 예측
  const output = model.predict(tf.tensor2d([5], [1, 1]));
  output.print(); // 예: 11 근처 출력
}

trainModel();
