import * as React from 'react';
import { HttpRequestContainer, DisplayHttpError } from '../utils/http_container';
import { DeploymentTable } from '../components/deployment_table';


export const DeploymentsList = () => (
  <HttpRequestContainer url='/api/ListDeployments' method='get'>
    {
      ({data, isLoading, error}) => {
        let activeDeploymentCount = 0;
        let deploymentDisplay;
        if (isLoading) {
          return <div>Loading...</div>
        }
        if (error) {
          return <DisplayHttpError error={error} />
        }
        if (data && data.deployments) {
          activeDeploymentCount = data.deployments.length;
          deploymentDisplay = (
            <DeploymentTable deployments={data.deployments} />
          )
        } else {
          deploymentDisplay = (
            <a href='https://docs.bentoml.org/en/latest' target='_blank'>
              Learn more about managing model serving deployments with BentoML
            </div>
          );
        }

        return (
          <div>
            <h2>Active Deployments: {activeDeploymentCount}</h2>
            {deploymentDisplay}
          </div>
        );
      }
    }
  </HttpRequestContainer>
);
