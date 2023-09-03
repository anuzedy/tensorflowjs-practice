const path = require('path');

module.exports = {
  entry: './deep.js',
  output: {
    path: path.resolve(__dirname, 'dist/js'),
    filename: 'bundle2.js'
  },
  devtool: 'source-map',
  mode: 'development'
};