import path from "path";
import morgan from "morgan";
import { Request, Response } from "express";
import express from "express";
import { bentoml } from "./generated/bentoml_grpc";
import { createYataiClient } from "./yatai_client";
import { getLogger } from "./logger";

const logger = getLogger();

const createRoutes = (app, yataiClient) => {
  app.get("/api/ListBento", async (req: Request, res: Response) => {
    const requestQuery: any = Object.assign({}, req.query);
    if (req.query.limit && typeof req.query.limit == "string") {
      requestQuery.limit = Number(req.query.limit);
    }
    if (req.query.offset && typeof req.query.offset == "string") {
      requestQuery.offset = Number(req.query.offset);
    }
    let verifyError = bentoml.ListBentoRequest.verify(requestQuery);
    if (verifyError) {
      logger.error({ request: "ListBento", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.ListBentoRequest.create(requestQuery);
    try {
      const result = await yataiClient.listBento(requestMessage);
      logger.info({
        request: "ListBento",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({ request: "ListBento", error: JSON.stringify(error) });
      return res.status(500).json(error);
    }
  });

  app.get("/api/GetBento", async (req: Request, res: Response) => {
    let verifyError = bentoml.GetBentoRequest.verify(req.query);
    if (verifyError) {
      logger.error({ request: "GetBento", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.GetBentoRequest.create(req.query);
    try {
      const result = await yataiClient.getBento(requestMessage);
      logger.info({
        request: "GetBento",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({ request: "GetBento", error: JSON.stringify(error) });
      return res.status(500).json(error);
    }
  });

  app.get("/api/GetDeployment", async (req: Request, res: Response) => {
    let verifyError = bentoml.GetDeploymentRequest.verify(req.query);
    if (verifyError) {
      logger.error({ request: "GetDeployment", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.GetDeploymentRequest.create(req.query);
    try {
      const result = await yataiClient.getDeployment(requestMessage);
      logger.info({
        request: "GetDeployment",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({ request: "GetDeployment", error: JSON.stringify(error) });
      return res.status(500).json(error);
    }
  });

  app.get("/api/ListDeployments", async (req: Request, res: Response) => {
    const requestQuery: any = Object.assign({}, req.query);
    if (req.query.limit && typeof req.query.limit == "string") {
      requestQuery.limit = Number(req.query.limit);
    }
    let verifyError = bentoml.ListDeploymentsRequest.verify(requestQuery);
    if (verifyError) {
      logger.error({ request: "ListDeployments", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.ListDeploymentsRequest.create(requestQuery);
    try {
      const result = await yataiClient.listDeployments(requestMessage);
      logger.info({
        request: "ListDeployments",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({ request: "ListDeployment", error: JSON.stringify(error) });
      return res.status(500).json(error);
    }
  });

  app.post("/api/DeleteDeployment", async (req: Request, res: Response) => {
    let verifyError = bentoml.DeleteDeploymentRequest.verify(req.body);
    if (verifyError) {
      logger.error({ request: "DeleteDeployment", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.DeleteDeploymentRequest.create(req.body);
    try {
      const result = await yataiClient.deleteDeployment(requestMessage);
      logger.info({
        request: "DeleteDeployment",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({
        request: "DeleteDeployment",
        error: JSON.stringify(error),
      });
      return res.status(500).json(error);
    }
  });

  app.post("/api/DeleteBento", async (req: Request, res: Response) => {
    let verifyError = bentoml.DangerouslyDeleteBentoRequest.verify(req.body);
    if (verifyError) {
      logger.error({ request: "DeleteBento", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.DangerouslyDeleteBentoRequest.create(req.body);
    try {
      const result = await yataiClient.dangerouslyDeleteBento(requestMessage);
      logger.info({
        request: "DeleteBento",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({ request: "DeleteBento", error: JSON.stringify(error) });
      return res.status(500).json(error);
    }
  });

  app.post("/api/ApplyDeployment", async (req: Request, res: Response) => {
    let verifyError = bentoml.ApplyDeploymentRequest.verify(req.body);
    if (verifyError) {
      logger.error({ request: "ApplyDeployment", error: verifyError });
      return res.status(400).json({ error: verifyError });
    }
    let requestMessage = bentoml.ApplyDeploymentRequest.create(req.body);
    try {
      const result = await yataiClient.applyDeployment(requestMessage);
      logger.info({
        request: "ApplyDeployment",
        data: requestMessage,
        result: result,
      });
      if (result.status.status_code != 0) {
        return res.status(400).json({ error: result.status.error_message });
      }
      return res.status(200).json(result);
    } catch (error) {
      logger.error({
        request: "ApplyDeployment",
        error: JSON.stringify(error),
      });
      return res.status(500).json(error);
    }
  });
};

export const getExpressApp = (grpcAddress: string | null, baseURL: string) => {
  const app = express();
  const cookieParser = require('cookie-parser');
  app.use(cookieParser());
  app.use(function (req, res, next) {
    var cookie = req.cookies.cookieName;
    if (cookie === undefined) {
      //var baseURL="/yatai/";
      res.cookie('baseURLCookie',baseURL, { maxAge: 900000, httpOnly: false });
      console.log('cookie created successfully');
    } else {
      console.log('cookie exists', cookie);
    } 
    next(); // <-- important!
  });

  app.use(express.json());
  app.use(
    morgan("combined", {
      stream: { write: (message) => logger.info(message.trim()) },
    })
  );
  const yataiClient = createYataiClient(grpcAddress);
  createRoutes(app, yataiClient);

  app.get("/*", (req, res) => {
    if (/.js$|.css$|.png$/.test(req.path)) {
      let directory = req.path.split("/").slice(-2, -1);
      let filename = req.path.split("/").pop();
      res.sendFile(path.join(__dirname, `../dist/client/static/${directory}/${filename}`));
    } else {
      res.sendFile(path.join(__dirname, "../dist/client/index.html"));
    }
  });
  
  app.use(express.static(path.join(__dirname, "../dist/client")));
  return app;
};
