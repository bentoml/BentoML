import * as React from "react";
import { Link } from "react-router-dom";
import { Table, Column, Cell } from "@blueprintjs/table";

import { displayTimeInFromNowFormat } from "../utils/index";
import { HttpRequestContainer } from '../utils/http_container'
import ConfigurationTable from '../components/DeploymentDetail/ConfigurationTable';
import DeploymentApisTable from '../components/DeploymentDetail/ApisTable';
import InfoTable from '../components/DeploymentDetail/InfoTable';
import { Button } from '@blueprintjs/core';


const DeploymentError = () => {
  return <div>if error display it</div>;
};


export const DeploymentDetails = props => {
  const params = props.match.params;
  return (
    <HttpRequestContainer
      url="/api/GetDeployment"
      params={{ deployment_name: params.name, namespace: params.namespace }}
    >
      {(data, isLoading, error) => {
        if (isLoading) {
          return <div>Loading...</div>
        }
        if (error) {
          return <div>error</div>;
        }
        let detailDisplay;

        if (data.data && data.data.deployment) {
          const deployment = data.data.deployment;
          detailDisplay = (
            <div>
              <h1>
                Deployment: {deployment.name} <Button>{deployment.state.state}</Button>
              </h1>
              {/* <DeploymentError /> */}
              <InfoTable deployment={deployment} />
              <ConfigurationTable spec={deployment.spec} />
              <DeploymentApisTable deployment={deployment} />
            </div>
          );
        } else {
          detailDisplay = <div>grpc error</div>;
        }
        return (
          <div>
            {detailDisplay}
          </div>
        );
      }}
    </HttpRequestContainer>
  );
};
