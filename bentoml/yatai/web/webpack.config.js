const path = require('path');
const spawnSync = require('child_process').spawnSync;

const webpack = require('webpack');
const { CheckerPlugin } = require('awesome-typescript-loader')

// Patch fs with graceful-fs for CopyWebpackPlugin to correctly copy node_modules
// in order to avoid "Error: EMFILE: too many open files, open.." in large project
const fs = require('fs');
const gracefulFs = require('graceful-fs');
gracefulFs.gracefulify(fs);

const rootDir = fs.realpathSync(process.cwd());

const nodeModulesPath = path.resolve(rootDir, 'node_modules');
const nodeModules = fs.readdirSync(nodeModulesPath).reduce((acc, mod) => {
  if (mod !== '.bin') {
    // eslint-disable-next-line no-param-reassign
    acc[mod] = `commonjs ${mod}`;
  }
  return acc;
}, {});

module.exports = (settings) => {
  // <option>=<default-value> (overwriting with 'settings' param)
  const {
    // Don't attempt to continue if there are any errors.
    bail=true,
    // Generate sourcemap file
    sourcemap=true,
    // webpack mode https://webpack.js.org/concepts/mode/
    mode='development',
  } = settings || {};

  return {
    mode: mode,
    bail: bail,
    target: 'node',
    node: {
      console: true,
      global: false,
      process: false,
      Buffer: false,
      __filename: false,
      __dirname: false,
    },
    entry: path.resolve(rootDir, 'src/index.ts'),
    output: {
      path: path.resolve(rootDir, 'dist'),
      filename: 'bundle.js',
      libraryTarget: "commonjs2",
    },
    externals: {
      express: 'commonjs express',
      protobufjs: 'commonjs protobufjs',
    },
    resolve: {
      extensions: ['.ts', '.js', '.tsx'],
      modules: [
        "node_modules"
      ],
    },
    devtool: sourcemap ? 'source-map' : false,
    module: {
      rules: [
        {
          test: /\.ts|\.tsx$/,
          include: [path.resolve(rootDir, 'src')],
          loader: 'awesome-typescript-loader',
        },
      ],
    },
    plugins: [
      new CheckerPlugin(),
    ],
    stats: {
      // Config for minimal console.log mess.
      assets: false,
      colors: true,
      version: false,
      hash: false,
      timings: false,
      chunks: false,
      chunkModules: false
    },
  };
};
