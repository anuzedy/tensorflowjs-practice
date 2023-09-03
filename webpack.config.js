const path = require('path');

module.exports = {
  entry: './classification.js',
  output: {
    path: path.resolve(__dirname, 'dist/js'),
    filename: 'bundle3.js'
  },
  devtool: 'source-map',
  mode: 'development'
};