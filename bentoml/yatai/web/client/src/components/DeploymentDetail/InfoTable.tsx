import * as React from 'react';
import { TableContainer, TableHeader, Row, Cell } from '../../ui/Table';
import { displayTimeInFromNowFormat } from '../../utils';
import { Link } from 'react-router-dom';


const infoList = ["Created at", "Updated at", "BentoService", "Endpoint"];

const InfoTable = ({deployment}) => {
  let endpointValues = 'Not Available';
  if (deployment.state.state == 'RUNNING' && deployment.state.info_json) {
    const infoJson = JSON.parse(deployment.state.info_json);
    endpointValues = infoJson.endpoints.join("\n");
  }
  return (
    <div>
      <h2>Info</h2>
      <TableContainer>
        <Row>
          <Cell>Created at</Cell>
          <Cell>
            {displayTimeInFromNowFormat(Number(deployment.created_at.seconds))}
          </Cell>
        </Row>
        <Row>
          <Cell>Updated at</Cell>
          <Cell>
            {displayTimeInFromNowFormat(Number(deployment.last_updated_at.seconds))}
          </Cell>
        </Row>
        <Row>
          <Cell>BentoService</Cell>
          <Cell>
            <Link
              to={`/repository/${deployment.spec.bento_name}/${deployment.spec.bento_version}`}
            >
              {`${deployment.spec.bento_name}:${deployment.spec.bento_version}`}
            </Link>
          </Cell>
        </Row>
        <Row>
          <Cell>Endpoints</Cell>
          <Cell>{endpointValues}</Cell>
        </Row>
      </TableContainer>
    </div>
  )
};

export default InfoTable;