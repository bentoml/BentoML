import * as React from 'react';
import * as moment from 'moment';
import { Link } from 'react-router-dom';
import { displayTimeInFromNowFormat } from '../utils/index';
import { TableContainer, TableHeader, Row, Cell } from '../ui/Table';


export const DeploymentTable = (props) => {
  const {deployments} = props;

  return (
    <TableContainer>
      <TableHeader>
        <Cell>Name</Cell>
        <Cell>Namespace</Cell>
        <Cell>Platform</Cell>
        <Cell>BentoService</Cell>
        <Cell>Status</Cell>
        <Cell>Age</Cell>
        <Cell>Last updated At</Cell>
        <Cell></Cell>
      </TableHeader>
      {
        deployments.map((deployment, i) => {
          const lastUpdatedAt = moment.unix(Number(deployment.last_updated_at.seconds))
            .format('MM/DD/YYYY HH:mm:ss Z');
          return (
            <Row key={i}>
              <Cell maxWidth={150} >{deployment.name}</Cell>
              <Cell maxWidth={150} >{deployment.namespace}</Cell>
              <Cell maxWidth={150} >{deployment.spec.operator}</Cell>
              <Cell maxWidth={500} >
                {`${deployment.spec.bento_name}:${deployment.spec.bento_version}`}
              </Cell>
              <Cell maxWidth={150} >{deployment.state.state}</Cell>
              <Cell maxWidth={150} >
                {displayTimeInFromNowFormat(Number(deployment.created_at.seconds))}
              </Cell>
              <Cell maxWidth={200} >{lastUpdatedAt}</Cell>
              <Cell maxWidth={150} >
                <Link to={`/deployments/${deployment.namespace}/${deployment.name}`}>
                  Detail
                </Link>
              </Cell>
            </Row>
          )
        })
      }
    </TableContainer>
  )
};