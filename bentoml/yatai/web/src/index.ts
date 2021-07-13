import { createLogger } from "./logger";
import { getExpressApp } from "./server";

const args = process.argv.slice(2);

const grpcServerAddress = args[0];
const port = args[1];
const base_log_path = args[2];
const base_url = args[3];
const prometheus_address = args[4];

createLogger(base_log_path);
const app = getExpressApp(grpcServerAddress, base_url, prometheus_address);

app.listen(port, () => console.log(`Running at http://localhost:${port}`));
