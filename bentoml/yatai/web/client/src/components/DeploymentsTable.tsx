import * as React from "react";
import * as moment from "moment";
import { Link } from "react-router-dom";

import { displayTimeInFromNowFormat } from "../utils/index";
import Table from "../ui/Table";

const DEPLOYMENTS_TABLE_HEADERS = [
  "Name",
  "Namespace",
  "Platform",
  "BentoService",
  "Status",
  "Age",
  "Last updated at",
  ""
];
const DEPLOYMENTS_TABLE_RATIO = [3, 2, 2, 5, 1, 2, 4, 1];

const DeploymentsTable = props => {
  const { deployments } = props;
  const parsedDeployments = deployments.map(deployment => {
    const lastUpdatedAt = moment
      .unix(Number(deployment.last_updated_at.seconds)).toDate().toLocaleString()

    return [
      deployment.name,
      deployment.namespace,
      deployment.spec.operator,
      `${deployment.spec.bento_name}:${deployment.spec.bento_version}`,
      deployment.state.state,
      displayTimeInFromNowFormat(Number(deployment.created_at.seconds)),
      lastUpdatedAt,
      <Link to={`/deployments/${deployment.namespace}/${deployment.name}`}>
        Detail
      </Link>
    ];
  });

  return (
    <Table
      content={parsedDeployments}
      ratio={DEPLOYMENTS_TABLE_RATIO}
      header={DEPLOYMENTS_TABLE_HEADERS}
    />
  );
};

export default DeploymentsTable;
