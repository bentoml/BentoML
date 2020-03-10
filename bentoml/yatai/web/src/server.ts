import path from 'path';
import { Request, Response } from "express";
import express from 'express'
import * as grpc from 'grpc';
import { bentoml } from './generated/bentoml_grpc';
import { createYataiClient } from './yatai_client';

const createRoutes = (app, yataiClient) => {
  app.get('/api/ListBento', async(req: Request, res: Response) => {
    let verifyError = bentoml.ListBentoRequest.verify(req.query);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.ListBentoRequest.create(req.query)
    let result = await yataiClient.listBento(requestMessage).then(response => response);
    return res.status(200).json(result);
  });

  app.get('/api/GetBento', async(req: Request, res: Response) => {
    let verifyError = bentoml.GetBentoRequest.verify(req.query);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.GetBentoRequest.create(req.query)
    let result = await yataiClient.getBento(requestMessage)
      .then(response => response);
    return res.status(200).json(result);
  });

  app.get('/api/GetDeployment', async(req: Request, res: Response) => {
    let verifyError = bentoml.GetDeploymentRequest.verify(req.query);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.GetDeploymentRequest.create(req.query)
    let result = await yataiClient.getDeployment(requestMessage)
      .then(response => response);
    return res.status(200).json(result);
  });

  app.get('/api/ListDeployments', async(req: Request, res: Response) => {
    let verifyError = bentoml.ListDeploymentsRequest.verify(req.query);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.ListDeploymentsRequest.create(req.query)
    let result = await yataiClient.listDeployments(requestMessage)
      .then(response => response);
    return res.status(200).json(result);
  });

  app.post('/api/DeleteDeployment', async(req: Request, res: Response) => {
    let verifyError = bentoml.DeleteDeploymentRequest.verify(req.body);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.DeleteDeploymentRequest.create(req.body)
    let result = await yataiClient.deleteDeployment(requestMessage)
      .then(response => response);
    return res.status(200).json(result);
  });

  app.post('/api/DeleteBento', async(req: Request, res: Response) => {
    let verifyError = bentoml.DangerouslyDeleteBentoRequest.verify(req.body);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.DangerouslyDeleteBentoRequest.create(req.body)
    let result = await yataiClient.dangerouslyDeleteBento(requestMessage)
      .then(response => response);
    return res.status(200).json(result);
  });

  app.post('/api/ApplyDeployment', async(req: Request, res: Response) => {
    let verifyError = bentoml.ApplyDeploymentRequest.verify(req.body);
    if (verifyError) {
      return res.status(400).json({error: verifyError})
    }
    let requestMessage = bentoml.ApplyDeploymentRequest.create(req.body)
    let result = await yataiClient.applyDeployment(requestMessage)
      .then(response => response);
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

  return app;
};