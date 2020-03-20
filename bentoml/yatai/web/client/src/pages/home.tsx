import * as React from 'react';
import { FetchContainer } from '../utils/index';
import { DeploymentTable } from '../components/deployment_table';
import { BentoTable } from '../components/bento_table';

export const Home = () => (
  <div>
    <div>
      <FetchContainer url='/api/ListDeployments' method='get'>
        {
          ({data, error}) => {
            let deploymentDisplay;
            let activeDeploymentCounts = 0;
            if (data && data.deployments) {
              deploymentDisplay = <DeploymentTable deployments={data.deployments} />;
              activeDeploymentCounts = data.deployments.length;
            } else {
              deploymentDisplay = (
                <div>
                  Learn about managing model serving deployments with BentoML
                </div>
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
      </FetchContainer>
    </div>
    <div>
      <h2>Latest Models</h2>
      <div>
        <FetchContainer url='/api/ListBento' method='get' params={{limit: 5}}>
          {
            ({data, error}) => {
              if (data && data.bentos) {
                return (
                  <BentoTable bentos={data.bentos} />
                );
              } else {
                return (<div>ok</div>)
              }
            }
          }
        </FetchContainer>
      </div>
    </div>
  </div>
);