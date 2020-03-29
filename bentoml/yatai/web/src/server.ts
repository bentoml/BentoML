import path from 'path';
import { Request, Response } from "express";
import express from 'express'
import { bentoml } from './generated/bentoml_grpc';
import { createYataiClient } from './yatai_client';
import { getLogger } from './logger';

const logger = getLogger();

const createRoutes = (app, yataiClient) => {
  app.get('/api/ListBento', async(req: Request, res: Response) => {
    if (req.query.limit && typeof(req.query.limit) == 'string') {
      req.query.limit = Number(req.query.limit)
    }
    if (req.query.offset && typeof(req.query.offset) == 'string') {
      req.query.offset = Number(req.query.offset)
    }
    let verifyError = bentoml.ListBentoRequest.verify(req.query);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.ListBentoRequest.create(req.query)
    let result = await yataiClient.listBento(requestMessage)
    logger.info({request: 'ListBento', data: requestMessage, result: result});
    if (result.status.status_code != 0) {
      return res.status(500).json({error: result.status.error_message});
    } else {
      return res.status(200).json(result);
    }
  });

  app.get('/api/GetBento', async(req: Request, res: Response) => {
    let verifyError = bentoml.GetBentoRequest.verify(req.query);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.GetBentoRequest.create(req.query)
    let result = await yataiClient.getBento(requestMessage);
    logger.info({request: 'GetBento', data: requestMessage, result: result});
    return res.status(200).json(result);
  });

  app.get('/api/GetDeployment', async(req: Request, res: Response) => {
    let verifyError = bentoml.GetDeploymentRequest.verify(req.query);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.GetDeploymentRequest.create(req.query)
    let result = await yataiClient.getDeployment(requestMessage);
    logger.info({request: 'GetDeployment', data: requestMessage, result: result});
    return res.status(200).json(result);
  });

  app.get('/api/ListDeployments', async(req: Request, res: Response) => {
    if (req.query.limit && typeof(req.query.limit) == 'string') {
      req.query.limit = Number(req.query.limit)
    }
    let verifyError = bentoml.ListDeploymentsRequest.verify(req.query);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.ListDeploymentsRequest.create(req.query)
    let result = await yataiClient.listDeployments(requestMessage);
    logger.info({request: 'ListDeployments', data: requestMessage, result: result});
    return res.status(200).json(result);
  });

  app.post('/api/DeleteDeployment', async(req: Request, res: Response) => {
    let verifyError = bentoml.DeleteDeploymentRequest.verify(req.body);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.DeleteDeploymentRequest.create(req.body)
    let result = await yataiClient.deleteDeployment(requestMessage);
    logger.info({request: 'DeleteDeployment', data: requestMessage, result: result});
    return res.status(200).json(result);
  });

  app.post('/api/DeleteBento', async(req: Request, res: Response) => {
    let verifyError = bentoml.DangerouslyDeleteBentoRequest.verify(req.body);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.DangerouslyDeleteBentoRequest.create(req.body)
    let result = await yataiClient.dangerouslyDeleteBento(requestMessage);
    logger.info({request: 'DeleteBento', data: requestMessage, result: result});
    return res.status(200).json(result);
  });

  app.post('/api/ApplyDeployment', async(req: Request, res: Response) => {
    let verifyError = bentoml.ApplyDeploymentRequest.verify(req.body);
    if (verifyError) {
      logger.error({error: verifyError});
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.ApplyDeploymentRequest.create(req.body)
    let result = await yataiClient.applyDeployment(requestMessage);
    logger.info({request: 'ApplyDeployment', data: requestMessage, result: result});
    return res.status(200).json(result);
  });
};

export const getExpressApp = (grpcAddress: string|null) => {
  const app = express()

  app.use(express.json())
  app.use(
    express.static(path.join(__dirname, '../dist/client'))
  )
  const yataiClient = createYataiClient(grpcAddress);
  createRoutes(app, yataiClient);

  app.get('/*', (req, res) => {
    if (/.js$|.css$/.test(req.path)) {
      let filename = req.path.split('/').pop();
      res.sendFile(path.join(__dirname, `../dist/client/${filename}`));
    } else {
      res.sendFile(path.join(__dirname, '../dist/client/index.html'));
    }
  })
  return app;
};
