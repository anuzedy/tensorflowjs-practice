const path = require('path');

module.exports = {
  entry: './tf.js',
  output: {
    path: path.resolve(__dirname, 'dist/js'),
    filename: 'bundle.js'
  },
  devtool: 'source-map',
  mode: 'development'
};