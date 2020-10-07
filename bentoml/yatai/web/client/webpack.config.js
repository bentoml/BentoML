const path = require("path");


module.exports = {
  // webpack is only used for production build in BentoML
  mode: "production", // "production" | "development" | "none"
  bail: true, // Don't attempt to continue if there are any errors.
  node: {
    console: true,
    global: false,
    process: false,
    Buffer: false,
    __filename: false,
    __dirname: false,
  },
  entry: path.resolve(__dirname, "src/index.tsx"),
  output: {
    path: path.resolve(__dirname, "dist/client"),
    filename: "bundle.js",
    libraryTarget: "commonjs2",
  },
  resolve: {
    extensions: [".ts", ".js", ".tsx", '.jsx'],
    modules: ["node_modules", path.resolve(__dirname), 'src'],
  },
  devtool: false,
  module: {
    rules: [
      {
        test: /\.ts|\.tsx$/,
        include: [path.resolve(__dirname, "src")],
        loader: "ts-loader",
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader']
      },
      {
        test: /\.eot(\?.*)?$/,
        loader: 'file-loader?name=fonts/[hash].[ext]'
      },
      {
        test: /\.(woff|woff2)(\?.*)?$/,
        loader: 'file-loader?name=fonts/[hash].[ext]'
      },
      {
        test: /\.ttf(\?.*)?$/,
        loader: 'url-loader?limit=10000&mimetype=application/octet-stream&name=fonts/[hash].[ext]'
      },
      {
        test: /\.svg(\?.*)?$/,
        loader: 'url-loader?limit=10000&mimetype=image/svg+xml&name=fonts/[hash].[ext]'
      },
      {
        test: /\.(jpe?g|png|gif|jp2|webp)$/,
        loader: 'file-loader',
      },
    ],
  },
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
