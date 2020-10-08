const path = require("path");

const { CheckerPlugin } = require("awesome-typescript-loader");

module.exports = {
  // webpack is only used for production build in BentoML
  mode: "production", // "production" | "development" | "none"
  target: "node",
  bail: true, // Don't attempt to continue if there are any errors.
  node: {
    console: true,
    global: false,
    process: false,
    Buffer: false,
    __filename: false,
    __dirname: false,
  },
  entry: path.resolve(__dirname, "src/index.ts"),
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bundle.js",
    libraryTarget: "commonjs2",
  },
  externals: {
    express: "commonjs express",
    protobufjs: "commonjs protobufjs",
  },
  resolve: {
    extensions: [".ts", ".js", ".tsx"],
    modules: ["node_modules"],
  },
  devtool: false,
  module: {
    rules: [
      {
        test: /\.ts|\.tsx$/,
        include: [path.resolve(__dirname, "src")],
        loader: "ts-loader",
      },
    ],
  },
  plugins: [new CheckerPlugin()],
  stats: {
    // Config for minimal console.log mess.
    assets: false,
    colors: true,
    version: false,
    hash: false,
    timings: false,
    chunks: false,
    chunkModules: false,
  },
};
