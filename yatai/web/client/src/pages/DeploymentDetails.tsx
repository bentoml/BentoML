import * as React from "react";

import HttpRequestContainer from "../utils/HttpRequestContainer";
import ConfigurationTable from "../components/DeploymentDetail/ConfigurationTable";
import DeploymentApisTable from "../components/DeploymentDetail/ApisTable";
import InfoTable from "../components/DeploymentDetail/InfoTable";
import DeploymentStatusTag from "../components/DeploymentDetail/DeploymentStatusTag";
import ErrorCard from "../components/DeploymentDetail/ErrorCard";

const DeploymentDetails = (props) => {
  const params = props.match.params;
  return (
    <HttpRequestContainer
      url="/api/GetDeployment"
      params={{ deployment_name: params.name, namespace: params.namespace }}
    >
      {(data) => {
        let detailDisplay;

        if (data.data && data.data.deployment) {
          const deployment = data.data.deployment;
          detailDisplay = (
            <div>
              <h1>
                Deployment: {deployment.name}{" "}
                <DeploymentStatusTag state={deployment.state.state} />
              </h1>
              <ErrorCard state={deployment.state} />
              <InfoTable deployment={deployment} />
              <ConfigurationTable spec={deployment.spec} />
              <DeploymentApisTable deployment={deployment} />
            </div>
          );
        } else {
          detailDisplay = <div>{JSON.stringify(data)}</div>;
        }
        return <div>{detailDisplay}</div>;
      }}
    </HttpRequestContainer>
  );
};

export default DeploymentDetails;
