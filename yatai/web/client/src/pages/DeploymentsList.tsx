import * as React from "react";
import * as lodash from "lodash";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import DeploymentsTable from "../components/DeploymentsTable";

const getActiveDeploymentCount = (deployments) => {
  return (
    lodash.filter(
      deployments,
      (deployment) =>
        deployment.state &&
        deployment.state.state &&
        deployment.state.state === "RUNNING"
    ).length || 0
  );
};

const DeploymentsList = (props) => {
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
          activeDeploymentCount = getActiveDeploymentCount(data.deployments);
          deploymentDisplay = (
            <DeploymentsTable deployments={data.deployments} />
          );
        } else {
          deploymentDisplay = (
            <a
              href="https://docs.bentoml.org/en/latest"
              target="_blank"
              rel="noopener noreferrer"
            >
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

export default DeploymentsList;
