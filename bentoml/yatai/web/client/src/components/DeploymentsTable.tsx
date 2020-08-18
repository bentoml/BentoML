import * as React from "react";
import { Link } from "react-router-dom";

import { displayTimeInFromNowFormat } from "../utils";
import Table from "../ui/Table";
import DeploymentStatusTag from "./DeploymentDetail/DeploymentStatusTag";

const DEPLOYMENTS_TABLE_HEADERS = [
  "Name",
  "Namespace",
  "Platform",
  "BentoService",
  "Status",
  "Age",
  "Last updated at",
  "",
];
const DEPLOYMENTS_TABLE_RATIO = [3, 2, 2, 5, 2, 1, 2, 1];

const DeploymentsTable = (props) => {
  const { deployments } = props;
  const parsedDeployments = deployments.map((deployment) => ({
    content: [
      deployment.name,
      deployment.namespace,
      deployment.spec.operator,
      `${deployment.spec.bento_name}:${deployment.spec.bento_version}`,
      <DeploymentStatusTag state={deployment.state.state} />,
      displayTimeInFromNowFormat(Number(deployment.created_at.seconds)),
      displayTimeInFromNowFormat(
        Number(deployment.last_updated_at.seconds),
        true
      ),
      <Link to={`/deployments/${deployment.namespace}/${deployment.name}`}>
        Detail
      </Link>,
    ],
    link: `/deployments/${deployment.namespace}/${deployment.name}`,
  }));

  return (
    <Table
      content={parsedDeployments}
      ratio={DEPLOYMENTS_TABLE_RATIO}
      header={DEPLOYMENTS_TABLE_HEADERS}
    />
  );
};

export default DeploymentsTable;
