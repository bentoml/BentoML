import { createLogger } from "./logger";
import { getExpressApp } from "./server";

const args = process.argv.slice(2);

const grpcServerAddress = args[0] || "localhost:50051";
const port = args[1] || "3000";
const base_log_path = args[2] || "./logs/default.txt";
const base_url = args[3] || ".";
const prometheus_address = args[4] || "localhost:3000/metrics";

createLogger(base_log_path);
const app = getExpressApp(grpcServerAddress, base_url, prometheus_address);

app.listen(port, () => console.log(`Running at http://localhost:${port}`));
