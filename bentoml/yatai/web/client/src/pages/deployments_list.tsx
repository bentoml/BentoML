import * as React from 'react';
import { HttpRequestContainer } from '../utils/http_container';
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
          return <div>Error: {JSON.stringify(error)}</div>
        }
        if (data && data.deployments) {
          deploymentDisplay = (
            <DeploymentTable deployments={data.deployments} />
          )
        } else {
          deploymentDisplay = (<div>Nop</div>)
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