import * as React from "react";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import DeploymentsTable from "../components/DeploymentsTable";
import BentoServiceTable from "../components/BentoServiceTable";

export const Home = () => (
  <div>
    <HttpRequestContainer url="/api/ListDeployments" method="get">
      {({ data }) => {
        let deploymentDisplay;
        let activeDeploymentCounts = 0;
        if (data && data.deployments) {
          deploymentDisplay = (
            <DeploymentsTable deployments={data.deployments} />
          );
          activeDeploymentCounts = data.deployments.length;
        } else {
          deploymentDisplay = (
            <a href="https://docs.bentoml.org/en/latest" target="_blank">
              Learn about managing model serving deployments with BentoML
              &#1f44b;
            </a>
          );
        }

        return (
          <div>
            <h2>Active Deployments: {activeDeploymentCounts}</h2>
            {deploymentDisplay}
          </div>
        );
      }}
    </HttpRequestContainer>
    <HttpRequestContainer
      url="/api/ListBento"
      method="get"
      params={{ limit: 5 }}
    >
      {({ data }) => {
        if (data && data.bentos) {
          return (
            <div>
              <h2>Latest Models</h2>
              <BentoServiceTable bentos={data.bentos} />
            </div>
          );
        } else {
          return (
            <div>
              <h2>No model found</h2>
              <a href="https://docs.bentoml.org/en/latest" target="_blank">
                Learn about packaging ML model for serving with BentoML &#1f517;
              </a>
            </div>
          );
        }
      }}
    </HttpRequestContainer>
  </div>
);
