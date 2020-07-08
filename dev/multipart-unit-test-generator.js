const http = require("http");
const fs = require("fs");
const path = require("path");

const server = http.createServer((req, res) => {
	if (req.method !== "POST") return;
	console.debug(`Delicious! Eating up a request with content-type ${req.headers['content-type']}`);
	const chunks = [];
	req.on("data", chunk => {
		chunks.push(chunk);
	});
	req.on("end", () => {
		bytes = Buffer.concat(chunks);
		fs.writeFileSync(path.resolve(__dirname, "../tests/multipart"), bytes);
		res.end("Thanks, that was delicious! Your multipart file is located at tests/multipart");
	});
});
console.log("My mouth is open on port 8000, and you can feed me with multipart-unit-test.html");
server.listen(8000);
