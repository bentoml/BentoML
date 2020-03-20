import * as React from 'react';
import { FetchContainer } from '../utils/index';


const DeploymentInfo = () => {
  return (
    <div>
      <h3>Info</h3>
      info
    </div>
  )
};


const DeploymentApis = () => {
  return (
    <div>
      <h3>APIs</h3>
      apis
    </div>
  )
};


const DeploymentEvents = () => {
  return (
    <div>
      <h3>Events</h3>
      events
    </div>
  )
};


const DeploymentError = () => {
  return (
    <div>
      if error display it
    </div>
  )
}


export const DeploymentDetails = (props) => {
  const params = props.match.params;
  return (
    <FetchContainer
      url='/api/GetDeployment'
      params={{deployment_name: params.name, namespace: params.namespace}}
    >
      {
        (data, error) => {
          console.log(data, error)
          let detailDisplay;
          if (error) {
            return (<div>error</div>)
          }

          if (data.data && data.data.deployment) {
            detailDisplay = (
              <div>
                <DeploymentError />
                <DeploymentInfo />
                <DeploymentApis />
                <DeploymentEvents />
              </div>
            );
          } else {
            detailDisplay = (
              <div>grpc error</div>
            )
          }
          return (
            <div>
              <div>breadcrumb</div>
              {detailDisplay}
            </div>
          );
        }
      }
    </FetchContainer>
  )
};