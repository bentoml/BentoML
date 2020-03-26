import * as React from "react";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import DeploymentsTable from "../components/DeploymentsTable";

export const DeploymentsList = props => {
  const params = props.match.params;
  let requestParams;
  if (params.namespace) {
    requestParams = { namespace: params.namespace };
  }
  return (
    <HttpRequestContainer
      url="/api/ListDeployments"
      method="get"
      params={requestParams}
    >
      {({ data }) => {
        let activeDeploymentCount = 0;
        let deploymentDisplay;

        if (data && data.deployments) {
          activeDeploymentCount = data.deployments.length;
          deploymentDisplay = (
            <DeploymentsTable deployments={data.deployments} />
          );
        } else {
          deploymentDisplay = (
            <a href="https://docs.bentoml.org/en/latest" target="_blank">
              Learn more about managing model serving deployments with BentoML
            </a>
          );
        }

        return (
          <div>
            <h2>Active Deployments: {activeDeploymentCount}</h2>
            {deploymentDisplay}
          </div>
        );
      }}
    </HttpRequestContainer>
  );
};
