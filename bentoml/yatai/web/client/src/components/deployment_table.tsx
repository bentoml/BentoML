import * as React from 'react';
import { Column, Table, Cell } from "@blueprintjs/table";
import * as moment from 'moment';
import { Link } from 'react-router-dom';
import { displayTimeInFromNowFormat } from '../utils/index';


export const DeploymentTable = (props) => {
  const {deployments} = props;

  const renderName = (rowIndex: number) => (
    <Cell>{deployments[rowIndex].name}</Cell>
  );

  const renderNamespace = (rowIndex: number) => (
    <Cell>{deployments[rowIndex].namespace}</Cell>
  );

  const renderPlatform = (rowIndex: number) => {
    const deployment = deployments[rowIndex];
    const platformName = deployment.spec.operator;

    return (
      <Cell>{platformName}</Cell>
    )
  };

  const renderBentoServiceTag = (rowIndex: number) => {
    const deployment = deployments[rowIndex];
    const bentoTag = `${deployment.spec.bento_name}:${deployment.spec.bento_version}`;
    return (
      <Cell>{bentoTag}</Cell>
    )
  };

  const renderStatus = (rowIndex: number) => {
    const deployment = deployments[rowIndex];
    const deploymentStatus = deployment.state.state;
    return (
      <Cell>{deploymentStatus}</Cell>
    )
  };

  const renderAge = (rowIndex: number) => {
    const deployment = deployments[rowIndex];

    return (
      <Cell>{displayTimeInFromNowFormat(Number(deployment.created_at.seconds))}</Cell>
    )
  };

  const renderLastUpdatedAt = (rowIndex: number) => {
    const deployment = deployments[rowIndex];
    const lastUpdatedAt = moment.unix(Number(deployment.last_updated_at.seconds))
      .format('MM/DD/YYYY HH:mm:ss Z');
    return (
      <Cell>{lastUpdatedAt}</Cell>
    )
  };

  const renderActions = (rowIndex: number) => {
    const deployment = deployments[rowIndex];

    return (
      <Cell>
        <Link to={`deployments/${deployment.namespace}/${deployment.name}`}>
          Detail
        </Link>
      </Cell>
    )
  }

  return (
    <Table numRows={deployments.length}>
      <Column name="Name" cellRenderer={renderName}/>
      <Column name="Namespace" cellRenderer={renderNamespace}/>
      <Column name="Platform" cellRenderer={renderPlatform}/>
      <Column name="BentoService" cellRenderer={renderBentoServiceTag} />
      <Column name='Status' cellRenderer={renderStatus} />
      <Column name='Age' cellRenderer={renderAge} />
      <Column name='Last updated at' cellRenderer={renderLastUpdatedAt} />
      <Column cellRenderer={renderActions} />
    </Table>
  );
};