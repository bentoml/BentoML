import * as React from "react";
import { Link } from "react-router-dom";
import { Tag, Intent } from '@blueprintjs/core';

import { displayTimeInFromNowFormat } from "../utils/index";
import { HttpRequestContainer, DisplayHttpError } from '../utils/http_container'
import ConfigurationTable from '../components/DeploymentDetail/ConfigurationTable';
import DeploymentApisTable from '../components/DeploymentDetail/ApisTable';
import InfoTable from '../components/DeploymentDetail/InfoTable';


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
          return <DisplayHttpError error={error} />
        }
        let detailDisplay;

        if (data.data && data.data.deployment) {
          const deployment = data.data.deployment;
          let statusColor;
          switch (deployment.state.state) {
            case 'RUNNING':
            case 'SUCCESSED':
              statusColor = Intent.SUCCESS;
              break;
            case 'FAILED':
            case 'ERROR':
            case 'CRASH_LOOP_BACK_OFF':
              statusColor = Intent.DANGER;
              break;
            default:
              statusColor = Intent.NONE;
          }

          const statusTag = <Tag intent={statusColor}>{deployment.state.state}</Tag>;
          detailDisplay = (
            <div>
              <h1>
                Deployment: {deployment.name} {statusTag}
              </h1>
              {/* <DeploymentError /> */}
              <InfoTable deployment={deployment} />
              <ConfigurationTable spec={deployment.spec} />
              <DeploymentApisTable deployment={deployment} />
            </div>
          );
        } else {
        detailDisplay = <div>{JSON.stringify(data)}</div>;
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
