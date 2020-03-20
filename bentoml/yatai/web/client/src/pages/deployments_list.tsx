import * as React from 'react';
import { FetchContainer } from '../utils/index';
import { DeploymentTable } from '../components/deployment_table';


export const DeploymentsList = () => (
  <FetchContainer url='/api/ListDeployments' method='get'>
    {
      ({data, error}) => {
        let activeDeploymentCount = 0;
        let deploymentDisplay;
        if (data && data.Deployments) {
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
  </FetchContainer>
);