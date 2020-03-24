import * as React from 'react';
import { HttpRequestContainer } from '../utils/http_container';
import { DeploymentTable } from '../components/deployment_table';
import { BentoTable } from '../components/bento_service_table';

export const Home = () => (
  <div>
    <div>
      <HttpRequestContainer url='/api/ListDeployments' method='get'>
        {
          ({data, isLoading, error}) => {
            if (isLoading) {
              return <div>Loading...</div>
            }
            if (error) {
              return <div>Error: {JSON.stringify(error)}</div>
            }
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
      </HttpRequestContainer>
    </div>
    <div>
      <h2>Latest Models</h2>
      <div>
        <HttpRequestContainer url='/api/ListBento' method='get' params={{limit: 5}}>
          {
            ({data, isLoading, error}) => {
              if (isLoading) {
                return <div>Loading...</div>
              }
              if (error) {
                return <div>Error: {JSON.stringify(error)}</div>
              }
              if (data && data.bentos) {
                return (
                  <BentoTable bentos={data.bentos} />
                );
              } else {
                return (<div>grpc error</div>);
              }
            }
          }
        </HttpRequestContainer>
      </div>
    </div>
  </div>
);