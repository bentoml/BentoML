import * as React from 'react';
import { HttpRequestContainer, DisplayHttpError } from '../utils/http_container';
import { DeploymentTable } from '../components/deployment_table';
import { BentoTable } from '../components/bento_service_table';
import { Link } from 'react-router-dom';

export const Home = () => (
  <div>
    <HttpRequestContainer url='/api/ListDeployments' method='get'>
      {
        ({data, isLoading, error}) => {
          if (isLoading) {
            return <div>Loading...</div>
          }
          if (error) {
            return <DisplayHttpError error={error} />
          }
          let deploymentDisplay;
          let activeDeploymentCounts = 0;
          if (data && data.deployments) {
            deploymentDisplay = <DeploymentTable deployments={data.deployments} />;
            activeDeploymentCounts = data.deployments.length;
          } else {
            deploymentDisplay = (
              <a href='https://docs.bentoml.org/en/latest' target="_blank">
                Learn about managing model serving deployments with BentoML &#1f44b;
              </a>
            );
          }

          return (
            <div>
              <h2>Active Deployments: {activeDeploymentCounts}</h2>
              {deploymentDisplay}
            </div>
          )
        }
      }
    </HttpRequestContainer>
    <HttpRequestContainer url='/api/ListBento' method='get' params={{limit: 5}}>
      {
        ({data, isLoading, error}) => {
          if (isLoading) {
            return <div>Loading...</div>
          }
          if (error) {
            return <DisplayHttpError error={error} />
          }
          if (data && data.bentos) {
            return (
              <div>
                <h2>Latest Models</h2>
                <BentoTable bentos={data.bentos} />
              </div>
            );
          } else {
            return (
              <div>
                <h2>No model found</h2>
                <a href='https://docs.bentoml.org/en/latest' target='_blank'>
                  Learn about packaging ML model for serving with BentoML &#1f517;
                </a>
              </div>
            )
          }
        }
      }
    </HttpRequestContainer>
  </div>
);
